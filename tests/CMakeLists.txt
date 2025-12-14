# Tests for SyntraLine++

# Unit tests executable
add_executable(syntra_tests
    unit/test_core.cpp
)

target_link_libraries(syntra_tests
    PRIVATE
        syntra_core
)

target_include_directories(syntra_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)