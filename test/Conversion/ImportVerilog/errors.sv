// RUN: circt-translate --import-verilog --verify-diagnostics %s

// expected-error @+1 {{expected ';'}}
module Foo 4;
endmodule

// expected-note @+1 {{expanded from macro 'FOO'}}
`define FOO input
// expected-note @+1 {{expanded from macro 'BAR'}}
`define BAR `FOO
// expected-error @+1 {{expected identifier}}
module Foo(`BAR);
endmodule
