#pragma once

#include <string>

#include "syntra/diagnostics.hpp"
#include "syntra/ir.hpp"
#include "syntra/backend_pytorch.hpp"
#include "syntra/backend_jax.hpp"

namespace syntra {

/// Which runtime backend we want to target when emitting a script.
enum class RuntimeBackendKind {
    Pytorch,
    Jax,
};

/// Options for script emission.
struct BackendRuntimeOptions {
    RuntimeBackendKind backendKind {RuntimeBackendKind::Pytorch};
    std::string experimentName; // optional: which experiment to run
};

/// High-level coordinator that picks a concrete backend (PyTorch, JAX, ...)
/// and emits a runnable script string.
///
/// This keeps main.cpp simpler and centralizes backend selection logic.
class BackendRuntime {
public:
    explicit BackendRuntime(DiagnosticsEngine& diag)
        : diag_(diag) {}

    /// Emit a script for the given module and options.
    /// Returns the script as a single string. On hard errors, returns empty.
    std::string emitScript(const PipelineModule& module,
                           const BackendRuntimeOptions& opts);

private:
    DiagnosticsEngine& diag_;
};

} // namespace syntra