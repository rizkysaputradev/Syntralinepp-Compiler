# Core SyntraLine++ compiler sources (excluding main.cpp)
set(SYNTRA_CORE_SRC
    config/syntra_config.cpp
    diag/diagnostics.cpp
    lexer/lexer.cpp
    parser/parser.cpp
    ast/ast.cpp
    sema/sema.cpp
    ir/ir_pipeline.cpp
    ir/ir_execution.cpp
    ir/ir_json.cpp
    optimizer/optimizer.cpp
    pass_manager/pass_manager.cpp
    backend/backend_pytorch.cpp
    backend/backend_jax.cpp 
    runtime/backend_runtime.cpp
)

add_library(syntra_core STATIC
    ${SYNTRA_CORE_SRC}
)

target_include_directories(syntra_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Main CLI executable
add_executable(syntra
    main.cpp
)

target_link_libraries(syntra
    PRIVATE syntra_core
)

if (SYNTRA_ENABLE_ASAN)
    if (NOT MSVC)
        target_compile_options(syntra_core PRIVATE -fsanitize=address)
        target_link_libraries(syntra_core PRIVATE -fsanitize=address)

        target_compile_options(syntra PRIVATE -fsanitize=address)
        target_link_libraries(syntra PRIVATE -fsanitize=address)
    endif ()
endif ()
