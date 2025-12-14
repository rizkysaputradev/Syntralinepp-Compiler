#pragma once

#include <memory>
#include <vector>

#include "syntra/optimizer.hpp"
#include "syntra/ir.hpp"
#include "syntra/diagnostics.hpp"

namespace syntra {

/// Simple pass manager that owns and runs pipeline passes in sequence.
class PassManager {
public:
    PassManager() = default;

    /// Add a pass to the pipeline.
    void addPass(std::unique_ptr<PipelinePass> pass);

    /// Add a reasonable default sequence of passes for v0.1.
    void addDefaultPasses();

    /// Run all passes in order on the module.
    void run(PipelineModule& module, DiagnosticsEngine& diag);

private:
    std::vector<std::unique_ptr<PipelinePass>> passes_;
};

} // namespace syntra