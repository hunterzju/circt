// RUN: circt-opt -hw-memory-sim %s | FileCheck %s

hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]

//CHECK-LABEL: @complex
hw.module @complex(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant true
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_2_4_0_0(ro_addr_0: %c0_i4: i4, ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1, 
     rw_wdata_0: %data0: i16,rw_wmask_0: %true: i1,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16, wo_mask_0: %true: i1) -> (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}

hw.module @complexMultiBit(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant 1 : i2
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi(ro_addr_0: %c0_i4: i4, ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1,
     rw_wdata_0: %data0: i16,rw_wmask_0: %true: i2,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16, wo_mask_0: %true: i2) -> (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}
//CHECK-LABEL: @simple
hw.module @simple(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant true
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_0_1_0_0( ro_addr_0: %c0_i4: i4,ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1, 
     rw_wdata_0: %data0: i16, rw_wmask_0: %true: i1,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16, wo_mask_0: %true: i1) -> 
     (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}

//CHECK-LABEL: @WriteOrderedSameClock
hw.module @WriteOrderedSameClock(%clock: i1, %w0_addr: i4, %w0_en: i1, %w0_data: i8, %w0_mask: i1, %w1_addr: i4, %w1_en: i1, %w1_data: i8, %w1_mask: i1) {
  hw.instance "memory"
    @FIRRTLMemOneAlways(wo_addr_0: %w0_addr: i4, wo_en_0: %w0_en: i1,
      wo_clock_0: %clock: i1, wo_data_0: %w0_data: i8, wo_mask_0: %w0_mask: i1,
      wo_addr_1: %w1_addr: i4, wo_en_1: %w1_en: i1, wo_clock_1: %clock: i1,
       wo_data_1: %w1_data: i8,wo_mask_1: %w1_mask: i1) -> ()
  hw.output
}

//CHECK-LABEL: @WriteOrderedDifferentClock
hw.module @WriteOrderedDifferentClock(%clock: i1, %clock2: i1, %w0_addr: i4, %w0_en: i1, %w0_data: i8, %w0_mask: i1, %w1_addr: i4, %w1_en: i1, %w1_data: i8, %w1_mask: i1) {
  hw.instance "memory"
    @FIRRTLMemTwoAlways(wo_addr_0: %w0_addr: i4, wo_en_0: %w0_en: i1,
      wo_clock_0: %clock: i1, wo_data_0: %w0_data: i8, wo_mask_0: %w0_mask: i1,
      wo_addr_1: %w1_addr: i4, wo_en_1: %w1_en: i1, wo_clock_1: %clock2: i1,
      wo_data_1: %w1_data: i8, wo_mask_1: %w1_mask: i1) -> ()
  hw.output
}

hw.module.generated @FIRRTLMem_1_1_1_16_10_0_1_0_0, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %rw_wmask_0: i1,  %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i1) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 0 : i32}

//CHECK-LABEL: @FIRRTLMem_1_1_1_16_10_0_1_0_0
//CHECK:       %Memory0 = sv.reg  : !hw.inout<uarray<10xi16>>
//CHECK-NEXT:  %[[rslot:.+]] = sv.array_index_inout %Memory0[%ro_addr_0]
//CHECK-NEXT:  %[[read1:.+]] = sv.read_inout %[[rslot]]
//CHECK-NEXT:  %[[read:.+]] = comb.concat %[[read1]] : i16
//CHECK-NEXT:  %[[x:.+]] = sv.constantX
//CHECK-NEXT:  %[[readres:.+]] = comb.mux %ro_en_0, %[[read]], %[[x]]
//CHECK-NEXT:  %[[rw_wmask_0:.+]] = comb.extract %rw_wmask_0 from 0 : (i1) -> i1
//CHECK-NEXT:  %[[rw_wdata_0:.+]] = comb.extract %rw_wdata_0 from 0 : (i16) -> i16
//CHECK-NEXT:  %[[rwtmp:.+]] = sv.wire
//CHECK-NEXT:  %[[rwres:.+]] = sv.read_inout %[[rwtmp]]
//CHECK-NEXT:  %false = hw.constant false
//CHECK-NEXT:  %[[rwrcondpre:.+]] = comb.icmp eq %rw_wmode_0, %false
//CHECK-NEXT:  %[[rwrcond:.+]] = comb.and %rw_en_0, %[[rwrcondpre]]
//CHECK-NEXT:  %[[rwslot:.+]] = sv.array_index_inout %Memory0[%rw_addr_0]
//CHECK-NEXT:  %[[v11:.+]] = sv.read_inout %10 : !hw.inout<i16>
//CHECK-NEXT:  %[[rwdata:.+]] = comb.concat %[[v11]] : i16
//CHECK-NEXT:  %[[x2:.+]] = sv.constantX
//CHECK-NEXT:  %[[rwdata2:.+]] = comb.mux %[[rwrcond]], %[[rwdata]], %[[x2]]
//CHECK-NEXT:  sv.assign %[[rwtmp]], %[[rwdata2:.+]]
//CHECK-NEXT:    sv.alwaysff(posedge %rw_clock_0)  {
//CHECK-NEXT:      %[[rwwcondpre:.+]] = comb.and %[[rw_wmask_0]], %rw_wmode_0
//CHECK-NEXT:      %[[rwwcond:.+]] = comb.and %rw_en_0, %[[rwwcondpre]]
//CHECK-NEXT:      sv.if %[[rwwcond]]  {
//CHECK-NEXT:        sv.passign %[[rwslot]], %[[rw_wdata_0]]
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:  %[[v14:.+]] = comb.extract %wo_mask_0 from 0 : (i1) -> i1
//CHECK-NEXT:  %[[v15:.+]] = comb.extract %wo_data_0 from 0 : (i16) -> i16
//CHECK-NEXT:  sv.alwaysff(posedge %wo_clock_0)  {
//CHECK-NEXT:    %[[wcond:.+]] = comb.and %wo_en_0, %[[v14]]
//CHECK-NEXT:    sv.if %[[wcond]]  {
//CHECK-NEXT:      %[[wslot:.+]] = sv.array_index_inout %Memory0[%wo_addr_0]
//CHECK-NEXT:      sv.passign %[[wslot]], %[[v15]]
//CHECK-NEXT:    }
//CHECK-NEXT:  }
//CHECK-NEXT:  hw.output %[[readres]], %[[rwres]]

hw.module.generated @FIRRTLMem_1_1_1_16_10_2_4_0_0, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %rw_wmask_0: i1,  %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i1) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeClockIDs = [], writeLatency = 4 : ui32, writeUnderWrite = 0 : i32}

//CHECK-LABEL: @FIRRTLMem_1_1_1_16_10_2_4_0_0
//COM: This produces a lot of output, we check one field's pipeline
//CHECK:         %Memory0 = sv.reg  : !hw.inout<uarray<10xi16>>
//CHECK:         sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %0, %ro_en_0 : i1
//CHECK-NEXT:    }
//CHECK-NEXT:    %1 = sv.read_inout %0 : !hw.inout<i1>
//CHECK-NEXT:    %2 = sv.reg  : !hw.inout<i1>
//CHECK-NEXT:    sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %2, %1 : i1
//CHECK-NEXT:    }
//CHECK-NEXT:    %3 = sv.read_inout %2 : !hw.inout<i1>
//CHECK-NEXT:    %4 = sv.reg  : !hw.inout<i4>
//CHECK-NEXT:    sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %4, %ro_addr_0 : i4
//CHECK-NEXT:    }
//CHECK-NEXT:    %5 = sv.read_inout %4 : !hw.inout<i4>
//CHECK-NEXT:    %6 = sv.reg  : !hw.inout<i4>
//CHECK-NEXT:    sv.alwaysff(posedge %ro_clock_0)  {
//CHECK-NEXT:      sv.passign %6, %5 : i4
//CHECK-NEXT:    }
//CHECK-NEXT:    %7 = sv.read_inout %6 : !hw.inout<i4>
//CHECK-NEXT:    %8 = sv.array_index_inout %Memory0[%7] : !hw.inout<uarray<10xi16>>, i4

hw.module.generated @FIRRTLMemOneAlways, @FIRRTLMem( %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,%wo_data_0: i8, %wo_mask_0: i1, %wo_addr_1: i4,  %wo_en_1: i1, %wo_clock_1: i1, %wo_data_1: i8, %wo_mask_1: i1) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [0 : i32, 0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

//CHECK-LABEL: @FIRRTLMemOneAlways
//CHECK-COUNT-1:  sv.alwaysff
//CHECK-NOT:      sv.alwaysff

hw.module.generated @FIRRTLMemTwoAlways, @FIRRTLMem( %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,%wo_data_0: i8, %wo_mask_0: i1, %wo_addr_1: i4,  %wo_en_1: i1, %wo_clock_1: i1, %wo_data_1: i8, %wo_mask_1: i1) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [0 : i32, 1 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

//CHECK-LABEL: @FIRRTLMemTwoAlways
//CHECK-COUNT-2:  sv.alwaysff
//CHECK-NOT:      sv.alwaysff


  hw.module.generated @FIRRTLMem_1_1_0_32_16_1_1_0_1_a, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i4) -> (R0_data: i32) attributes {depth = 16 : i64, maskGran = 8 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 32 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  hw.module @memTestFoo(%clock: i1, %rAddr: i4, %rEn: i1, %wAddr: i4, %wEn: i1, %wMask: i4, %wData: i32) -> (rData: i32) attributes {firrtl.moduleHierarchyFile = #hw.output_file<"testharness_hier.json", excludeFromFileList>} {
    %memory.R0_data = hw.instance "memory" @FIRRTLMem_1_1_0_32_16_1_1_0_1_a(R0_addr: %rAddr: i4, R0_en: %rEn: i1, R0_clk: %clock: i1, W0_addr: %wAddr: i4, W0_en: %wEn: i1, W0_clk: %clock: i1, W0_data: %wData: i32, W0_mask: %wMask: i4) -> (R0_data: i32)
    hw.output %memory.R0_data : i32
  }
  // CHECK-LABEL: hw.module @FIRRTLMem_1_1_0_32_16_1_1_0_1_a
  // CHECK-SAME: (%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i4) -> (R0_data: i32)
  // CHECK-NEXT:   %[[Memory0:.+]] = sv.reg  : !hw.inout<uarray<16xi8>>
  // CHECK-NEXT:   %[[Memory1:.+]] = sv.reg  : !hw.inout<uarray<16xi8>>
  // CHECK-NEXT:   %[[Memory2:.+]] = sv.reg  : !hw.inout<uarray<16xi8>>
  // CHECK-NEXT:   %[[Memory3:.+]] = sv.reg  : !hw.inout<uarray<16xi8>>
  // CHECK:        %[[v10:.+]] = sv.array_index_inout %[[Memory3]][%[[v3:.+]]] : !hw.inout<uarray<16xi8>>, i4
  // CHECK:        %[[v11:.+]] = sv.read_inout %[[v10]] : !hw.inout<i8>
  // CHECK:        %[[v8:.+]] = sv.array_index_inout %[[Memory2]][%[[v3]]] : !hw.inout<uarray<16xi8>>, i4
  // CHECK:        %[[v9:.+]] = sv.read_inout %[[v8]] : !hw.inout<i8>
  // CHECK:        %[[v6:.+]] = sv.array_index_inout %[[Memory1]][%[[v3]]] : !hw.inout<uarray<16xi8>>, i4
  // CHECK:        %[[v7:.+]] = sv.read_inout %[[v6]] : !hw.inout<i8>
  // CHECK:        %[[v4:.+]] = sv.array_index_inout %[[Memory0]][%[[v3]]] : !hw.inout<uarray<16xi8>>, i4
  // CHECK:        %[[v5:.+]] = sv.read_inout %[[v4]] : !hw.inout<i8>
  // CHECK-NEXT:   %[[v12:.+]] = comb.concat %[[v11]], %[[v9]], %[[v7]], %[[v5]] : i8, i8, i8, i8
  // CHECK-NEXT:   %[[x_i32:.+]] = sv.constantX : i32
  // CHECK-NEXT:   %[[v13:.+]] = comb.mux %[[v1:.+]], %[[v12]], %[[x_i32]] : i32
  // CHECK-NEXT:   %[[v14:.+]] = comb.extract %W0_mask from 0 : (i4) -> i1
  // CHECK-NEXT:   %[[v15:.+]] = comb.extract %W0_data from 0 : (i32) -> i8
  // CHECK-NEXT:   %[[v16:.+]] = comb.extract %W0_mask from 1 : (i4) -> i1
  // CHECK-NEXT:   %[[v17:.+]] = comb.extract %W0_data from 8 : (i32) -> i8
  // CHECK-NEXT:   %[[v18:.+]] = comb.extract %W0_mask from 2 : (i4) -> i1
  // CHECK-NEXT:   %[[v19:.+]] = comb.extract %W0_data from 16 : (i32) -> i8
  // CHECK-NEXT:   %[[v20:.+]] = comb.extract %W0_mask from 3 : (i4) -> i1
  // CHECK-NEXT:   %[[v21:.+]] = comb.extract %W0_data from 24 : (i32) -> i8
  // CHECK-NEXT:   sv.alwaysff(posedge %W0_clk)  {
  // CHECK-NEXT:     %[[v22:.+]] = comb.and %W0_en, %14 : i1
  // CHECK-NEXT:     sv.if %[[v22]]  {
  // CHECK-NEXT:       %[[v26:.+]] = sv.array_index_inout %[[Memory0]][%W0_addr] : !hw.inout<uarray<16xi8>>, i4
  // CHECK-NEXT:       sv.passign %[[v26]], %[[v15]] : i8

  hw.module.generated @FIRRTLMem_1_1_0_32_16_1_1_0_1_b, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i2) -> (R0_data: i32) attributes {depth = 16 : i64, maskGran = 16 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : ui32, width = 32 : ui32, writeClockIDs = [0 : i32], writeLatency = 3 : ui32, writeUnderWrite = 1 : i32}
  hw.module @memTestBar(%clock: i1, %rAddr: i4, %rEn: i1, %wAddr: i4, %wEn: i1, %wMask: i2, %wData: i32) -> (rData: i32) attributes {firrtl.moduleHierarchyFile = #hw.output_file<"testharness_hier.json", excludeFromFileList>} {
    %memory.R0_data = hw.instance "memory" @FIRRTLMem_1_1_0_32_16_1_1_0_1_b(R0_addr: %rAddr: i4, R0_en: %rEn: i1,
    R0_clk: %clock: i1, W0_addr: %wAddr: i4, W0_en: %wEn: i1, W0_clk: %clock: i1, W0_data: %wData: i32, W0_mask: %wMask:  i2) -> (R0_data: i32)
    hw.output %memory.R0_data : i32
  }
  // CHECK-LABEL:hw.module @FIRRTLMem_1_1_0_32_16_1_1_0_1_b
  // CHECK-SAME: (%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i2) -> (R0_data: i32)
  // CHECK:  %[[Memory0:.+]] = sv.reg  : !hw.inout<uarray<16xi16>>
  // CHECK:  %[[Memory1:.+]] = sv.reg  : !hw.inout<uarray<16xi16>>
  // CHECK:  %[[v10:.+]] = sv.array_index_inout %[[Memory1]][%[[v7:.+]]] : !hw.inout<uarray<16xi16>>, i4
  // CHECK:  %[[v11:.+]] = sv.read_inout
  // CHECK:  %[[v8:.+]] = sv.array_index_inout %[[Memory0]][%[[v7]]] : !hw.inout<uarray<16xi16>>, i4
  // CHECK:  %[[v9:.+]] = sv.read_inout
  // CHECK:  %[[v12:.+]] = comb.concat %[[v11]], %[[v9]] : i16, i16
  // CHECK:  %[[x_i32:.+]] = sv.constantX : i32
  // CHECK:  %[[v13:.+]] = comb.mux
  // CHECK:  sv.alwaysff(posedge %W0_clk)  {
  // CHECK:    sv.passign %[[v22:.+]], %W0_data : i32
  // CHECK:  }
  // CHECK:  %[[v23:.+]] = sv.read_inout %[[v22]] : !hw.inout<i32>
  // CHECK:  %[[v24:.+]] = sv.reg  : !hw.inout<i32>
  // CHECK:  sv.alwaysff(posedge %W0_clk)  {
  // CHECK:    sv.passign %[[v24:.+]], %[[v23:.+]] : i32
  // CHECK:  }
  // CHECK:  %[[v25:.+]] = sv.read_inout %[[v24]] : !hw.inout<i32>
  // CHECK:  sv.alwaysff(posedge %W0_clk)  {
  // CHECK:    sv.passign %[[v26:.+]], %W0_mask : i2
  // CHECK:  }
  // CHECK:  sv.alwaysff(posedge %W0_clk)  {
  // CHECK:    sv.passign %[[v28:.+]], %[[v27:.+]] : i2
  // CHECK:  }
  // CHECK:  %[[v29:.+]] = sv.read_inout %[[v28]] : !hw.inout<i2>
  // CHECK:  %[[v30:.+]] = comb.extract %[[v29]] from 0 : (i2) -> i1
  // CHECK:  %[[v31:.+]] = comb.extract %[[v25]] from 0 : (i32) -> i16
  // CHECK:  %[[v32:.+]] = comb.extract %[[v29]] from 1 : (i2) -> i1
  // CHECK:  %[[v33:.+]] = comb.extract %[[v25]] from 16 : (i32) -> i16
  // CHECK:  sv.alwaysff(posedge %W0_clk)  {
  // CHECK:    %[[v34:.+]] = comb.and %[[v21:.+]], %[[v30]] : i1
  // CHECK:    sv.if %[[v34]]  {
  // CHECK:      %[[v36:.+]] = sv.array_index_inout %[[Memory0]][%[[v17]]] : !hw.inout<uarray<16xi16>>, i4
  // CHECK:      sv.passign %[[v36]], %[[v31]] : i16
  // CHECK:    }
  // CHECK:    %[[v35:.+]] = comb.and %[[v21]], %[[v32]] : i1
  // CHECK:    sv.if %[[v35]]  {
  // CHECK:      %[[v36:.+]] = sv.array_index_inout %[[Memory1]][%[[v17]]] : !hw.inout<uarray<16xi16>>, i4
  // CHECK:      sv.passign %[[v36]], %[[v33]] : i16
  // CHECK:    }
  // CHECK:  }
  // CHECK:  hw.output %[[v13]] : i32
  // CHECK:}

hw.module.generated @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0:
i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %rw_wmask_0: i2,  %wo_addr_0: i4, %wo_en_0: i1,
%wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i2) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64,
numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32,maskGran = 8 :ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeClockIDs = [], writeLatency = 4 : ui32, writeUnderWrite = 0 : i32}

// CHECK-LABEL:  hw.module @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi
// CHECK-SAME: %ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1, %rw_addr_0: i4,
// CHECK-SAME: %rw_en_0: i1, %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,
// CHECK-SAME: %rw_wmask_0: i2, %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,
// CHECK-SAME: %wo_data_0: i16, %wo_mask_0: i2
// CHECK:    %[[Memory0:.+]] = sv.reg  : !hw.inout<uarray<10xi8>>
// CHECK:    %[[Memory1:.+]] = sv.reg  : !hw.inout<uarray<10xi8>>
// CHECK:    %[[v10:.+]] = sv.array_index_inout %[[Memory1]][%[[v7:.+]]] : !hw.inout<uarray<10xi8>>, i4
// CHECK:    %[[v11:.+]] = sv.read_inout %[[v10]] : !hw.inout<i8>
// CHECK:    %[[v8:.+]] = sv.array_index_inout %[[Memory0]][%[[v7]]] : !hw.inout<uarray<10xi8>>, i4
// CHECK:    %[[v9:.+]] = sv.read_inout %[[v8]] : !hw.inout<i8>
// CHECK:    %[[v12:.+]] = comb.concat %[[v11]], %[[v9]] : i8, i8
// CHECK:    sv.alwaysff(posedge %rw_clock_0)  {
// CHECK:      sv.passign %[[v38:.+]], %rw_wmask_0 : i2
// CHECK:    %[[v44:.+]] = comb.extract %[[v43:.+]] from 0 : (i2) -> i1
// CHECK:    %[[v45:.+]] = comb.extract %[[v37:.+]] from 0 : (i16) -> i8
// CHECK:    %[[v46:.+]] = comb.extract %[[v43]] from 1 : (i2) -> i1
// CHECK:    %[[v47:.+]] = comb.extract %[[v37]] from 8 : (i16) -> i8
// CHECK:    %[[v52:.+]] = sv.array_index_inout %[[Memory0]][%[[v19:.+]]] : !hw.inout<uarray<10xi8>>, i4
// CHECK:    %[[v54:.+]] = sv.array_index_inout %[[Memory1]][%[[v19:.+]]] : !hw.inout<uarray<10xi8>>, i4
// CHECK:    %[[v55:.+]] = sv.read_inout %[[v54]] : !hw.inout<i8>
// CHECK:    %[[v56:.+]] = comb.concat %[[v55]], %[[v53:.+]] : i8, i8
// CHECK:    sv.alwaysff(posedge %wo_clock_0)  {
// CHECK:      sv.passign %[[v70:.+]], %wo_data_0 : i16
// CHECK:    }
// CHECK:    sv.alwaysff(posedge %wo_clock_0)  {
// CHECK:      sv.passign %[[v76:.+]], %wo_mask_0 : i2
// CHECK:    }
// CHECK:    %[[v82:.+]] = comb.extract %[[v81:.+]] from 0 : (i2) -> i1
// CHECK:    %[[v83:.+]] = comb.extract %[[v75:.+]] from 0 : (i16) -> i8
// CHECK:    %[[v84:.+]] = comb.extract %[[v81]] from 1 : (i2) -> i1
// CHECK:    %[[v85:.+]] = comb.extract %[[v75]] from 8 : (i16) -> i8
// CHECK:    sv.alwaysff(posedge %wo_clock_0)  {
// CHECK:      %[[v86:.+]] = comb.and %[[v69:.+]], %[[v82]] : i1
// CHECK:      sv.if %[[v86]]  {
// CHECK:        %[[v88:.+]] = sv.array_index_inout %[[Memory0]][%[[v63:.+]]] : !hw.inout<uarray<10xi8>>, i4
// CHECK:        sv.passign %[[v88]], %[[v83]] : i8
// CHECK:      }
// CHECK:      sv.if %[[v87:.+]]  {
// CHECK:        %[[v88:.+]] = sv.array_index_inout %[[Memory1]][%[[v63]]] : !hw.inout<uarray<10xi8>>, i4
// CHECK:        sv.passign %[[v88]], %[[v85]] : i8
// CHECK:      }
// CHECK:    }
