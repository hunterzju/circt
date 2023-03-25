//===- VerilogToSV.h - Verilog frontend integration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Verilog frontend.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_VERILOGTOSV_H
#define CIRCT_CONVERSION_VERILOGTOSV_H

#include "circt/Support/LLVM.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class LocationAttr;
class TimingScope;
} // namespace mlir

namespace circt {

/// Parse files in a source manager as Verilog source code.
mlir::OwningOpRef<mlir::ModuleOp> verilog2SV(llvm::SourceMgr &sourceMgr,
                                             mlir::MLIRContext *context,
											 mlir::TimingScope &ts);

/// Register the `Verilog2SV` MLIR translation.
void registerVerilog2SVTranslation();

} // namespace circt

#endif // CIRCT_CONVERSION_VERILOGTOSV_H
