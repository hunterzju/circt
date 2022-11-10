// RUN: circt-translate --import-verilog --verify-diagnostics %s

module Foo;
    bit [3:0] x;
    assign x = 4;
endmodule

package Bar;
endpackage
