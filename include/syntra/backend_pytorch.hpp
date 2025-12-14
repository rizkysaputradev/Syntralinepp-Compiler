#pragma once

#include <string>

#include "syntra/diagnostics.hpp"
#include "syntra/ir.hpp"

namespace syntra {

/// Backend that lowers PipelineModule into a runnable PyTorch Python script.
///
/// v0.1 goals:
///   - Generate a single script containing:
///       * imports
///       * dataset stubs / loaders based on IRDataset fields
///       * simple model definition based on IRModel fields
///       * training / evaluation loops based on IRExperimentPipeline
///   - The script should run even if data paths are placeholders, so it is
///     usable as a template.
///
/// We intentionally keep this simple and explicit instead of trying to be
/// fully generic ML framework codegen in the first version.
class PytorchBackend {
  public:
    explicit PytorchBackend(DiagnosticsEngine &diag) : diag_(diag) {}

    /// Emit a complete Python script representing the given module.
    /// If `selectedExperimentName` is non-empty, the generated main()
    /// will run that experiment if it exists; otherwise it falls back
    /// to the first experiment.
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

    std::string getStringField(const std::vector<const NamedField *> &fields,
                               const std::string &name,
                               const std::string &defaultValue = "") const;

    std::string getNumericField(const std::vector<const NamedField *> &fields,
                                const std::string &name,
                                const std::string &defaultValue = "") const;

    std::string getBoolField(const std::vector<const NamedField *> &fields,
                             const std::string &name,
                             const std::string &defaultValue = "False") const;

    DiagnosticsEngine &diag_;
};

} // namespace syntra