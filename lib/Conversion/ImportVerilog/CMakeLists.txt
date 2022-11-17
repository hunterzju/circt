# slang uses exceptions
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

# Disable some compiler warnings cause by slang headers such that the
# `ImportVerilog` build doesn't spew out a ton of warnings that are not related
# to CIRCT.
if (MSVC)
  # No idea what to put here
else ()
  # slang has some classes with virtual funcs but non-virtual destructor.
  add_compile_options(-Wno-non-virtual-dtor)
  # some other warnings we've seen
  add_compile_options(-Wno-c++98-compat-extra-semi)
  add_compile_options(-Wno-ctad-maybe-unsupported)
endif ()

add_circt_translation_library(CIRCTImportVerilog
  ConvertRoot.cpp
  ImportVerilog.cpp

  LINK_LIBS PUBLIC
  CIRCTMoore
  MLIRTranslateLib
  PRIVATE
  slang::slang
)
