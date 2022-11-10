# slang uses exceptions
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

add_circt_translation_library(CIRCTImportVerilog
  ImportVerilog.cpp

  LINK_LIBS PUBLIC
  CIRCTMoore
  MLIRTranslateLib
  PRIVATE
  slang::slang
)
