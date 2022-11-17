//===- ConvertRoot.cpp - Slang AST to MLIR conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"

using namespace circt;
using namespace slang::syntax;
using namespace ImportVerilog;

using slang::SourceLocation;
using ConvertLocationFn = std::function<Location(SourceLocation)>;

namespace {
class RootConverter : public SyntaxVisitor<RootConverter> {
public:
  ModuleOp module;
  ConvertLocationFn convertLocation;
  LogicalResult result = success();

  RootConverter(ModuleOp module, ConvertLocationFn convertLocation)
      : module(module), convertLocation(convertLocation) {}

  void handle(const ModuleDeclarationSyntax &decl) {
    mlir::emitRemark(convertLocation(decl.header->moduleKeyword.location()))
        << "Found module " << decl.header->name.valueText();
  }
};
} // namespace

LogicalResult circt::ImportVerilog::convertSyntaxTree(
    slang::syntax::SyntaxTree &syntax, ModuleOp module,
    llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath) {
  RootConverter visitor(module, [&](SourceLocation loc) {
    return convertLocation(module.getContext(), syntax.sourceManager(),
                           getBufferFilePath, loc);
  });
  syntax.root().visit(visitor);
  return visitor.result;
}
