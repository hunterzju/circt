// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
  func @test() {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @test(
// CHECK-SAME:                         %[[VAL_0:.*]]: none, ...) -> none {
// CHECK:           %[[VAL_1:.*]]:5 = "handshake.memory"(%[[VAL_2:.*]]#0, %[[VAL_2]]#1, %[[VAL_3:.*]], %[[VAL_4:.*]]) {id = 0 : i32, ld_count = 2 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index, index) -> (f32, f32, none, none, none)
// CHECK:           %[[VAL_5:.*]]:2 = "handshake.fork"(%[[VAL_1]]#4) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_6:.*]]:2 = "handshake.fork"(%[[VAL_0]]) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_7:.*]]:3 = "handshake.fork"(%[[VAL_6]]#1) {control = true} : (none) -> (none, none, none)
// CHECK:           %[[VAL_8:.*]] = "handshake.join"(%[[VAL_7]]#2, %[[VAL_1]]#3) {control = true} : (none, none) -> none
// CHECK:           %[[VAL_9:.*]] = "handshake.constant"(%[[VAL_7]]#1) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_10:.*]] = "handshake.constant"(%[[VAL_7]]#0) {value = 10 : index} : (none) -> index
// CHECK:           %[[VAL_11:.*]]:2 = "handshake.fork"(%[[VAL_10]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_12:.*]], %[[VAL_3]] = "handshake.load"(%[[VAL_11]]#0, %[[VAL_1]]#0, %[[VAL_6]]#0) : (index, f32, none) -> (f32, index)
// CHECK:           %[[VAL_13:.*]] = "handshake.branch"(%[[VAL_8]]) {control = true} : (none) -> none
// CHECK:           %[[VAL_14:.*]] = "handshake.branch"(%[[VAL_9]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_15:.*]] = "handshake.branch"(%[[VAL_11]]#1) {control = false} : (index) -> index
// CHECK:           %[[VAL_16:.*]] = "handshake.branch"(%[[VAL_12]]) {control = false} : (f32) -> f32
// CHECK:           %[[VAL_17:.*]] = "handshake.mux"(%[[VAL_18:.*]]#2, %[[VAL_19:.*]], %[[VAL_15]]) : (index, index, index) -> index
// CHECK:           %[[VAL_20:.*]]:2 = "handshake.fork"(%[[VAL_17]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_21:.*]] = "handshake.mux"(%[[VAL_18]]#1, %[[VAL_22:.*]], %[[VAL_16]]) : (index, f32, f32) -> f32
// CHECK:           %[[VAL_23:.*]]:2 = "handshake.control_merge"(%[[VAL_24:.*]], %[[VAL_13]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_18]]:3 = "handshake.fork"(%[[VAL_23]]#1) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_25:.*]] = "handshake.mux"(%[[VAL_18]]#0, %[[VAL_26:.*]], %[[VAL_14]]) : (index, index, index) -> index
// CHECK:           %[[VAL_27:.*]]:2 = "handshake.fork"(%[[VAL_25]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_27]]#1, %[[VAL_20]]#1 : index
// CHECK:           %[[VAL_29:.*]]:4 = "handshake.fork"(%[[VAL_28]]) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = "handshake.conditional_branch"(%[[VAL_29]]#3, %[[VAL_20]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_31]]) : (index) -> ()
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = "handshake.conditional_branch"(%[[VAL_29]]#2, %[[VAL_21]]) {control = false} : (i1, f32) -> (f32, f32)
// CHECK:           "handshake.sink"(%[[VAL_33]]) : (f32) -> ()
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = "handshake.conditional_branch"(%[[VAL_29]]#1, %[[VAL_23]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = "handshake.conditional_branch"(%[[VAL_29]]#0, %[[VAL_27]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_37]]) : (index) -> ()
// CHECK:           %[[VAL_38:.*]] = "handshake.merge"(%[[VAL_36]]) : (index) -> index
// CHECK:           %[[VAL_39:.*]] = "handshake.merge"(%[[VAL_32]]) : (f32) -> f32
// CHECK:           %[[VAL_40:.*]]:2 = "handshake.fork"(%[[VAL_39]]) {control = false} : (f32) -> (f32, f32)
// CHECK:           %[[VAL_41:.*]] = "handshake.merge"(%[[VAL_30]]) : (index) -> index
// CHECK:           %[[VAL_42:.*]]:2 = "handshake.control_merge"(%[[VAL_34]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_43:.*]]:3 = "handshake.fork"(%[[VAL_42]]#0) {control = true} : (none) -> (none, none, none)
// CHECK:           %[[VAL_44:.*]]:2 = "handshake.fork"(%[[VAL_43]]#2) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_45:.*]] = "handshake.join"(%[[VAL_44]]#1, %[[VAL_5]]#1, %[[VAL_1]]#2) {control = true} : (none, none, none) -> none
// CHECK:           "handshake.sink"(%[[VAL_42]]#1) : (index) -> ()
// CHECK:           %[[VAL_46:.*]] = "handshake.constant"(%[[VAL_44]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_47:.*]] = arith.addi %[[VAL_38]], %[[VAL_46]] : index
// CHECK:           %[[VAL_48:.*]]:3 = "handshake.fork"(%[[VAL_47]]) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_49:.*]], %[[VAL_4]] = "handshake.load"(%[[VAL_48]]#2, %[[VAL_1]]#1, %[[VAL_43]]#1) : (index, f32, none) -> (f32, index)
// CHECK:           %[[VAL_50:.*]] = arith.addf %[[VAL_40]]#1, %[[VAL_49]] : f32
// CHECK:           %[[VAL_51:.*]] = "handshake.join"(%[[VAL_43]]#0, %[[VAL_5]]#0) {control = true} : (none, none) -> none
// CHECK:           %[[VAL_2]]:2 = "handshake.store"(%[[VAL_50]], %[[VAL_48]]#1, %[[VAL_51]]) : (f32, index, none) -> (f32, index)
// CHECK:           %[[VAL_22]] = "handshake.branch"(%[[VAL_40]]#0) {control = false} : (f32) -> f32
// CHECK:           %[[VAL_19]] = "handshake.branch"(%[[VAL_41]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_24]] = "handshake.branch"(%[[VAL_45]]) {control = true} : (none) -> none
// CHECK:           %[[VAL_26]] = "handshake.branch"(%[[VAL_48]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_52:.*]]:2 = "handshake.control_merge"(%[[VAL_35]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_52]]#1) : (index) -> ()
// CHECK:           handshake.return %[[VAL_52]]#0 : none
// CHECK:         }
// CHECK:       }

    %10 = memref.alloc() : memref<10xf32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %5 = memref.load %10[%c10] : memref<10xf32>
    br ^bb1(%c0 : index)
  ^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %1, %c10 : index
    cond_br %2, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    %c1 = arith.constant 1 : index
    %3 = arith.addi %1, %c1 : index
    %7 = memref.load %10[%3] : memref<10xf32>
    %8 = arith.addf %5, %7 : f32
    memref.store %8, %10[%3] : memref<10xf32>
    br ^bb1(%3 : index)
  ^bb3: // pred: ^bb1
    return
  }
