//===- ImportVerilog.cpp - Slang Verilog frontend integration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements bridging from the slang Verilog frontend to CIRCT dialects.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/text/SourceManager.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/SourceMgr.h"

using namespace circt;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

namespace {
/// A converter that can be plugged into a slang `DiagnosticEngine` as a client
/// that will map slang diagnostics to their MLIR counterpart and emit them.
class MlirDiagnosticClient : public slang::DiagnosticClient {
public:
  MlirDiagnosticClient(
      MLIRContext *context,
      std::function<StringRef(slang::BufferID)> getBufferFilePath)
      : context(context), getBufferFilePath(getBufferFilePath) {}

  void report(const slang::ReportedDiagnostic &diag) override {
    // Generate the primary MLIR diagnostic.
    auto &diagEngine = context->getDiagEngine();
    auto mlirDiag = diagEngine.emit(convertLocation(diag.location),
                                    getSeverity(diag.severity));
    mlirDiag << diag.formattedMessage;

    // Append the name of the option that can be used to control this
    // diagnostic.
    auto optionName = engine->getOptionName(diag.originalDiagnostic.code);
    if (!optionName.empty())
      mlirDiag << " [-W" << optionName << "]";

    // Write out macro expansions, if we have any, in reverse order.
    for (auto it = diag.expansionLocs.rbegin(); it != diag.expansionLocs.rend();
         it++) {
      auto &note = mlirDiag.attachNote(
          convertLocation(sourceManager->getFullyOriginalLoc(*it)));
      auto macro_name = sourceManager->getMacroName(*it);
      if (macro_name.empty())
        note << "expanded from here";
      else
        note << "expanded from macro '" << macro_name << "'";
    }
  }

  /// Convert a slang `SourceLocation` to an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc) const {
    if (loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
      // auto fileName = sourceManager.getFileName(loc);
      auto fileName = getBufferFilePath(loc.buffer());
      auto line = sourceManager->getLineNumber(loc);
      auto column = sourceManager->getColumnNumber(loc);
      return FileLineColLoc::get(context, fileName, line, column);
    }
    return UnknownLoc::get(context);
  }

  static mlir::DiagnosticSeverity
  getSeverity(slang::DiagnosticSeverity severity) {
    switch (severity) {
    case slang::DiagnosticSeverity::Fatal:
    case slang::DiagnosticSeverity::Error:
      return mlir::DiagnosticSeverity::Error;
    case slang::DiagnosticSeverity::Warning:
      return mlir::DiagnosticSeverity::Warning;
    case slang::DiagnosticSeverity::Ignored:
    case slang::DiagnosticSeverity::Note:
      return mlir::DiagnosticSeverity::Remark;
    }
    llvm_unreachable("all slang diagnostic severities should be handled");
    return mlir::DiagnosticSeverity::Error;
  }

private:
  MLIRContext *context;
  std::function<StringRef(slang::BufferID)> getBufferFilePath;
};
} // namespace

// Allow for slang::BufferID to be used as hash map keys.
namespace llvm {
template <>
struct DenseMapInfo<slang::BufferID> {
  static slang::BufferID getEmptyKey() { return slang::BufferID(); }
  static slang::BufferID getTombstoneKey() {
    return slang::BufferID(UINT32_MAX - 1, ""sv);
    // UINT32_MAX is already used by `BufferID::getPlaceholder`.
  }
  static unsigned getHashValue(slang::BufferID id) {
    return llvm::hash_value(id.getId());
  }
  static bool isEqual(slang::BufferID a, slang::BufferID b) { return a == b; }
};
} // namespace llvm

// Parse the specified Verilog inputs into the specified MLIR context.
mlir::OwningOpRef<mlir::ModuleOp> circt::importVerilog(SourceMgr &sourceMgr,
                                                       MLIRContext *context,
                                                       mlir::TimingScope &ts) {
  // Populate a slang source manager with the source files.
  // NOTE: This is a bit ugly since we're essentially copying the Verilog source
  // text in memory. At a later stage we might want to extend slang's
  // SourceManager such that it can contain non-owned buffers. This will do for
  // now.
  slang::SourceManager slangSM;
  SmallVector<slang::SourceBuffer> slangBuffers;

  // We keep a separate map from slang's buffers to the original MLIR file name
  // since slang's `SourceLocation::getFileName` returns a modified version that
  // is nice for human consumption (proximate paths, just file names, etc.), but
  // breaks MLIR's assumption that the diagnostics report the exact file paths
  // that appear in the `SourceMgr`. We use this separate map to lookup the
  // exact paths and use those for reporting.
  // See: https://github.com/MikePopoloski/slang/discussions/658
  SmallDenseMap<slang::BufferID, StringRef> bufferFilePaths;

  for (unsigned i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(i + 1);
    auto slangBuffer =
        slangSM.assignText(buffer->getBufferIdentifier(), buffer->getBuffer());
    slangBuffers.push_back(slangBuffer);
    bufferFilePaths.insert({slangBuffer.id, buffer->getBufferIdentifier()});
  }

  // Parse the input.
  auto parseTimer = ts.nest("Verilog parser");
  auto syntaxTree =
      slang::syntax::SyntaxTree::fromBuffers(slangBuffers, slangSM);
  parseTimer.stop();

  // Emit any diagnostics that got generated during parsing.
  slang::DiagnosticEngine diagEngine(slangSM);
  auto diagClient = std::make_shared<MlirDiagnosticClient>(
      context, [&](slang::BufferID id) { return bufferFilePaths.lookup(id); });
  diagEngine.addClient(diagClient);
  for (auto &diag : syntaxTree->diagnostics())
    diagEngine.issue(diag);
  if (diagEngine.getNumErrors() > 0)
    return {};

  // Traverse the parsed Verilog AST and map it to the equivalent CIRCT ops.
  context->loadDialect<moore::MooreDialect>();
  mlir::OwningOpRef<ModuleOp> module(
      ModuleOp::create(UnknownLoc::get(context)));
  // TODO: Do AST traversal.

  // Run the verifier on the constructed module to ensure it is clean.
  auto verifierTimer = ts.nest("Post-parse verification");
  if (failed(verify(*module)))
    return {};
  return module;
}

void circt::registerFromVerilogTranslation() {
  static mlir::TranslateToMLIRRegistration fromVerilog(
      "import-verilog", "import Verilog or SystemVerilog",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        return importVerilog(sourceMgr, context, ts);
      });
}
