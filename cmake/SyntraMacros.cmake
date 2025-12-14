# =============================================================================
# SyntraMacros.cmake
# =============================================================================
# Helper macros/functions for the SyntraLine++ project.
#
# These are intentionally small and self-contained. They are safe to include
# even if not used anywhere in the root CMakeLists.txt.
# =============================================================================

# -----------------------------------------------------------------------------
# syntra_enable_strict_warnings(target)
# -----------------------------------------------------------------------------
# Enable a reasonably strict warning set for a C++ target.
# Platform-specific flags are guarded with generator expressions / checks.
# -----------------------------------------------------------------------------
function(syntra_enable_strict_warnings target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
            -Wdouble-promotion
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# syntra_set_cxx_standard(target)
# -----------------------------------------------------------------------------
# Force C++20 (or newer if you later want C++23) consistently.
# -----------------------------------------------------------------------------
function(syntra_set_cxx_standard target)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
endfunction()

# -----------------------------------------------------------------------------
# syntra_add_test(target test_name)
# -----------------------------------------------------------------------------
# Simple wrapper for adding tests with a consistent naming scheme.
# Usage:
#   syntra_add_test(syntra_tests "unit_core")
# -----------------------------------------------------------------------------
function(syntra_add_test target test_name)
    if (NOT TARGET ${target})
        message(FATAL_ERROR "syntra_add_test: target '${target}' does not exist")
    endif()

    # This is a generic helper; you may already have tests() configured.
    # If you want, you can call this from your tests/CMakeLists.txt.
    add_test(
        NAME ${test_name}
        COMMAND ${target}
    )
endfunction()