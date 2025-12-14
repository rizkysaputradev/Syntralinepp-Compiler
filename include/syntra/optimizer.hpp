#pragma once

#include <string>

#include "syntra/ir.hpp"
#include "syntra/diagnostics.hpp"

namespace syntra {

/// Base class for IR-level pipeline passes.
class PipelinePass {
public:
    virtual ~PipelinePass() = default;

    /// Human-readable pass name.
    virtual const char* name() const = 0;

    /// Run the pass on the given module.
    virtual void run(PipelineModule& module, DiagnosticsEngine& diag) = 0;
};

/// If an experiment has no explicit `return { ... }`, we create a canonical
/// result mapping from the last evaluate(...) op's metrics:
///   metrics=[accuracy, loss] => return { accuracy=eval_var.accuracy, loss=eval_var.loss }
class CanonicalizeResultsPass : public PipelinePass {
public:
    const char* name() const override { return "CanonicalizeResultsPass"; }
    void run(PipelineModule& module, DiagnosticsEngine& diag) override;
};

/// Validate that experiment result fields refer to valid evaluation vars and
/// metrics (e.g. `test_run.accuracy` where `test_run` is an eval op var that
/// actually produced `accuracy`).
class ValidateMetricsPass : public PipelinePass {
public:
    const char* name() const override { return "ValidateMetricsPass"; }
    void run(PipelineModule& module, DiagnosticsEngine& diag) override;
};

} // namespace syntra