//===- circt-opt.cpp - The circt-opt driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-opt' tool, which is the circt analog of
// mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// Defined in the test directory, no public header.
namespace circt {
namespace test {
void registerAnalysisTestPasses();
void registerSchedulingTestPasses();
} // namespace test
} // namespace circt

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  circt::registerAllDialects(registry);
  circt::registerAllPasses();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  // Register test passes
  circt::test::registerAnalysisTestPasses();
  circt::test::registerSchedulingTestPasses();

  // Other command line options.
  circt::registerLoweringCLOptions();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "CIRCT modular optimizer driver", registry,
                        /*preloadDialectsInContext=*/false));
}
