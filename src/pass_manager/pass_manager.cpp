#include "syntra/pass_manager.hpp"

namespace syntra {

void PassManager::addPass(std::unique_ptr<PipelinePass> pass) {
    passes_.push_back(std::move(pass));
}

void PassManager::addDefaultPasses() {
    // Order matters: canonicalize first, then validate.
    addPass(std::make_unique<CanonicalizeResultsPass>());
    addPass(std::make_unique<ValidateMetricsPass>());
}

void PassManager::run(PipelineModule& module, DiagnosticsEngine& diag) {
    for (auto& pass : passes_) {
        if (!pass) continue;
        pass->run(module, diag);
        if (diag.hasErrors()) {
            // Stop running further passes if a pass introduced hard errors.
            return;
        }
    }
}

} // namespace syntra