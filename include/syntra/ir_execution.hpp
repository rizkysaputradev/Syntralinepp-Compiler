#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "syntra/ir.hpp"
#include "syntra/diagnostics.hpp"

namespace syntra {

/// A generic key/value metric result (e.g. accuracy, loss, etc.)
using MetricMap = std::unordered_map<std::string, double>;

/// Result of running a single experiment in the in-process runtime.
struct ExperimentRunResult {
    std::string experimentName;
    MetricMap metrics;        // e.g. { "accuracy": 0.93, "loss": 0.12 }
};

/// Simple in-process execution engine that "simulates" the pipeline IR.
/// It does NOT perform real ML training; instead, it:
///   - walks through train/evaluate ops,
///   - produces plausible placeholder metrics,
///   - wires them through ExperimentResultField mapping.
///
/// This is a backend-agnostic sanity checker for the IR and execution model.
/// Later, it can be extended to hook into actual C++ backends.
class PipelineExecutor {
public:
    explicit PipelineExecutor(DiagnosticsEngine& diag)
        : diag_(diag) {}

    /// Run the first experiment in the module and return its result.
    /// If there are no experiments, reports a diagnostic and returns an empty result.
    ExperimentRunResult runFirstExperiment(const PipelineModule& module);

    /// Run a specific experiment by name. If not found, reports a diagnostic
    /// and returns an empty result.
    ExperimentRunResult runExperimentByName(const PipelineModule& module,
                                            const std::string& name);

private:
    ExperimentRunResult runExperiment(const IRExperimentPipeline& exp);

    // Helpers
    MetricMap simulateTrain(const PipelineTrainOp& trainOp);
    MetricMap simulateEvaluate(const PipelineEvaluateOp& evalOp);

    DiagnosticsEngine& diag_;
};

} // namespace syntra