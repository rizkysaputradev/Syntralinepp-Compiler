#include "syntra/backend_runtime.hpp"

namespace syntra {

std::string BackendRuntime::emitScript(const PipelineModule& module,
                                       const BackendRuntimeOptions& opts) {
    switch (opts.backendKind) {
    case RuntimeBackendKind::Pytorch: {
        PytorchBackend backend(diag_);
        return backend.emitPythonScript(module, opts.experimentName);
    }
    case RuntimeBackendKind::Jax: {
        JaxBackend backend(diag_);
        return backend.emitPythonScript(module, opts.experimentName);
    }
    }

    diag_.error(SourceLocation{},
                "BackendRuntime::emitScript: unknown backend kind");
    return {};
}

} // namespace syntra