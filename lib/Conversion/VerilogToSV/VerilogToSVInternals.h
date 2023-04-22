//===- ImportVerilogInternals.h - Internal implementation details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_VERILOG2SV_VERILOG2SVINTERNALS_H
#define CONVERSION_VERILOG2SV_VERILOG2SVINTERNALS_H

#include "ASTSymbols.h"
#include "circt/Conversion/VerilogToSV.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Expression.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/expressions/AssertionExpr.h"
#include "slang/syntax/SyntaxNode.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/syntax/SyntaxVisitor.h"
#include "slang/text/SourceLocation.h"
#include "slang/text/SourceManager.h"

namespace circt {
namespace Verilog2SV {

Location convertLocation(
    MLIRContext *context, const slang::SourceManager &sourceManager,
    llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath,
    slang::SourceLocation loc);

LogicalResult convertSyntaxTree(
    slang::syntax::SyntaxTree &syntax, ModuleOp module,
    llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath);

LogicalResult
convertAST(const slang::ast::Symbol &symbol, ModuleOp module,
           const slang::SourceManager &srcMgr,
           llvm::function_ref<StringRef(slang::BufferID)> getBufferFilePath);

using ConvertLocationFn = std::function<Location(slang::SourceLocation)>;

/// Slang Visitor for SystemVerilog
class SVASTVisitor : public slang::ast::ASTVisitor<SVASTVisitor, true, true> {
public:
  /// The module is the container of the design.
  ModuleOp module;
  ConvertLocationFn convertLocation;
  LogicalResult result = success();

  SVASTVisitor(ModuleOp module, MLIRContext *ctx,
               ConvertLocationFn convertLocation)
      : module(module), convertLocation(convertLocation), context(ctx),
        builder(ctx) {}
  ~SVASTVisitor() = default;

#define HANDLE_AST_NODE(NodeName, NodeTy, RetTy)                               \
  RetTy handle(const slang::ast::NodeName##NodeTy &node);

#define HANDLE_AST_SYMBOL_VOID(NodeName) HANDLE_AST_NODE(NodeName, Symbol, void)

  // Handle statement
  // Handle expression
  // Handle symbol
  AST_SYMBOLKIND(HANDLE_AST_SYMBOL_VOID)

  // Invalid
  void visitInvalid(const slang::ast::Symbol &node) {}
  void visitInvalid(const slang::ast::BinsSelectExpr &node) {}
  void visitInvalid(const slang::ast::Expression &node) {}
  void visitInvalid(const slang::ast::Statement &node) {}
  void visitInvalid(const slang::ast::AssertionExpr &node) {}

private:
  MLIRContext *context = nullptr;
  OpBuilder builder;
};

} // namespace Verilog2SV
} // namespace circt

#endif // CONVERSION_VERILOG2SV_VERILOG2SVINTERNALS_H
