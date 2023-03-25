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
  add_compile_options(-Wno-cast-qual)
endif ()

add_compile_options(-fexceptions)
add_compile_options(-frtti)

message(STATUS "cmake cxx flags: ${CMAKE_CXX_FLAGS}")

add_circt_translation_library(CIRCTVerilog2SV
  ConvertRoot.cpp
  Verilog2SV.cpp

  LINK_LIBS PUBLIC
  CIRCTSV
  MLIRTranslateLib
  PRIVATE
  slang::slang
)

# slang need the -DDEBUG definition while the circt gives -D_DEBUG.
if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  add_compile_definitions(DEBUG)
endif ()
