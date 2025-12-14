# =============================================================================
# LLVMConfig.cmake
# =============================================================================
# Optional LLVM integration for SyntraLine++.
#
# Right now, SyntraLine++ uses its own custom IR and does not depend on LLVM.
# This file is prepared for a future phase where:
#   - you might want to lower SyntraLine++ IR to LLVM IR, or
#   - link against LLVM support libraries.
#
# This module is safe to include unconditionally; if LLVM is not installed,
# it will simply not expose LLVM_* variables.
# =============================================================================

# Try to find LLVM quietly. If not found, we do not error out.
find_package(LLVM QUIET)

if (LLVM_FOUND)
    message(STATUS "LLVM found: ${LLVM_PACKAGE_VERSION}")
    # LLVM provides a list of useful definitions and include dirs.
    message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
    message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")

    # Export a small set of variables for use in the main CMakeLists.txt
    set(SYNTRA_LLVM_FOUND TRUE CACHE BOOL "Whether LLVM was found for SyntraLine++")
    set(SYNTRA_LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}" CACHE STRING "LLVM include dirs")
    set(SYNTRA_LLVM_DEFINITIONS "${LLVM_DEFINITIONS}" CACHE STRING "LLVM definitions")

    # Example of how you might configure a target (for future use):
    #   target_include_directories(syntra_core PRIVATE ${SYNTRA_LLVM_INCLUDE_DIRS})
    #   target_compile_definitions(syntra_core PRIVATE ${SYNTRA_LLVM_DEFINITIONS})
else()
    set(SYNTRA_LLVM_FOUND FALSE CACHE BOOL "Whether LLVM was found for SyntraLine++")
    # No error: LLVM is optional at this stage.
endif()