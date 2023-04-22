#include "../VerilogToSVInternals.h"
#include "ASTSymbols.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include <memory>

using namespace circt;
using namespace Verilog2SV;
using namespace slang::ast;

namespace circt {
namespace Verilog2SV {

class RootSymConverter {
public:
  RootSymConverter(const slang::ast::RootSymbol &astNode, OpBuilder &builder,
                   Location &loc)
      : symbol(astNode), builder(builder), loc(loc) {}

  void convert();

private:
  const slang::ast::RootSymbol &symbol;
  OpBuilder &builder;
  Location &loc;
};

void RootSymConverter::convert() {
  mlir::emitRemark(loc) << "test root converter";
}

void SVASTVisitor::handle(const slang::ast::RootSymbol &symbol) {
  auto loc = convertLocation(symbol.location);
  std::unique_ptr<RootSymConverter> converter =
      std::make_unique<RootSymConverter>(symbol, builder, loc);
  converter->convert();
}

} // namespace Verilog2SV
} // namespace circt
