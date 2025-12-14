# ============================================================
# START
# ============================================================
cmake_minimum_required(VERSION 3.16)

project(SyntraLinePP
    VERSION 0.1.0
    DESCRIPTION "SyntraLine++ — Experimental Intelligence Language Compiler"
    LANGUAGES CXX C
)

# ============================================================
# Global configuration
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable more warnings by default
if (MSVC)
    add_compile_options(/W4 /permissive-)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif ()

# Optionally we can enable LTO / sanitizers later
option(SYNTRA_ENABLE_LTO "Enable link-time optimization" OFF)
option(SYNTRA_ENABLE_ASAN "Enable AddressSanitizer for debug builds" OFF)

# ============================================================
# Include directories
# ============================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================
# Subdirectories
# ============================================================
add_subdirectory(src)
add_subdirectory(runtime)
add_subdirectory(tests)

# ============================================================
# END
# ============================================================