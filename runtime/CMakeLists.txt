# Runtime support for SyntraLine++
# --------------------------------
# At this stage, this directory is a placeholder for future runtime
# components:
#   - Python helper scripts
#   - PyTorch/JAX utilities
#   - Syntra VM runtime
#
# For now, we don't build any concrete runtime library here. This file
# exists only to satisfy add_subdirectory(runtime) at the top level.

# You can later do things like:
# add_subdirectory(python)
# add_subdirectory(pytorch)
# add_subdirectory(jax)
# add_subdirectory(syntra_vm)

# ---------------------------------------------------------------------------
# Include Syntra helper modules
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(SyntraMacros)          # safe even if unused
include(LLVMConfig)            # optional: detects LLVM if available

