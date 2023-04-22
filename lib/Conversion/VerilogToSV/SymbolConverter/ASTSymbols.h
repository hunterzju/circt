#ifndef CONVERSION_VERILOGTOSV_ASTSYMBOL_H
#define CONVERSION_VERILOGTOSV_ASTSYMBOL_H

#include "slang/ast/Symbol.h"

using namespace slang::ast;

class Symbol;

#define AST_SYMBOLKIND(x)                                                      \
  x(Root) // x(CompilationUnit)       \
  // x(Instance)              \
  // x(InstanceBody)                                        \
      // TODO: add other symbols.

#define AST_SYMBOL_DECL(Kind) class Kind##Symbol;

AST_SYMBOLKIND(AST_SYMBOL_DECL)

// Declare for SymbolConveter class
namespace circt {
namespace Verilog2SV {
#define CONVERTER_DECL(Kind) class Kind##SymConverter;
AST_SYMBOLKIND(CONVERTER_DECL)

} // namespace Verilog2SV
} // namespace circt
#endif
