//===- ImportVerilogInternals.h - Internal implementation details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
#define CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H

#include "circt/Conversion/ImportVerilog.h"
#include "mlir/IR/BuiltinOps.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/syntax/SyntaxVisitor.h"
#include "slang/text/SourceManager.h"

namespace circt {
namespace ImportVerilog {

Location convertLocation(
    MLIRContext *context, const slang::SourceManager &sourceManager,
    llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath,
    slang::SourceLocation loc);

LogicalResult convertSyntaxTree(
    slang::syntax::SyntaxTree &syntax, ModuleOp module,
    llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath);

} // namespace ImportVerilog
} // namespace circt

#endif // CONVERSION_IMPORTVERILOG_IMPORTVERILOGINTERNALS_H
