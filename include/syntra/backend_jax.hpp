#pragma once

#include <string>

#include "syntra/diagnostics.hpp"
#include "syntra/ir.hpp"

namespace syntra {

/// Backend that lowers PipelineModule into a JAX/Flax-style Python script.
///
/// v0.1 goals:
///   - Emit a scaffolded script with:
///       * imports (jax, jax.numpy, typing)
///       * dataset stubs
///       * simple MLP-like model scaffold
///       * placeholder train/eval loops with TODO comments
///   - Clearly mark the script as a starting point, not a fully tuned JAX
///   system.
class JaxBackend {
  public:
    explicit JaxBackend(DiagnosticsEngine &diag) : diag_(diag) {}

    /// Emit a scaffold JAX-based Python script.
    /// If `selectedExperimentName` is non-empty, we will try to run that
    /// experiment in main(); otherwise we fall back to the first experiment.
    std::string
    emitPythonScript(const PipelineModule &module,
                     const std::string &selectedExperimentName = "");

  private:
    std::string emitHeader() const;
    std::string emitDatasetSection(const PipelineModule &module) const;
    std::string emitModelSection(const PipelineModule &module) const;
    std::string emitExperimentSection(const PipelineModule &module) const;
    std::string
    emitMainSection(const PipelineModule &module,
                    const std::string &selectedExperimentName) const;

    DiagnosticsEngine &diag_;
};

} // namespace syntra