#include "syntra/optimizer.hpp"

#include <unordered_map>
#include <unordered_set>

namespace syntra {

// ============================================================================
// CanonicalizeResultsPass
// ============================================================================

void CanonicalizeResultsPass::run(PipelineModule& module, DiagnosticsEngine& diag) {
    (void)diag; // currently unused; might be useful later

    for (auto& exp : module.experiments) {
        // If there are already explicit results, don't touch them.
        if (!exp.results.empty()) {
            continue;
        }

        // Find the last evaluate op for this experiment, if any.
        const PipelineOp* lastEval = nullptr;
        for (const auto& op : exp.ops) {
            if (op.kind == PipelineOpKind::Evaluate) {
                lastEval = &op;
            }
        }

        if (!lastEval) {
            // Nothing to canonicalize if there's no evaluate op.
            continue;
        }

        // Create result fields from the last eval op's metrics.
        const auto& evalOp = lastEval->eval;
        for (const auto& metricName : evalOp.metrics) {
            ExperimentResultField rf;
            rf.fieldName  = metricName;          // e.g. "accuracy"
            rf.sourceVar  = evalOp.varName;      // e.g. "test_run"
            rf.metricName = metricName;          // "accuracy"
            rf.loc        = lastEval->loc;       // reuse op location
            exp.results.push_back(std::move(rf));
        }
    }
}

// ============================================================================
// ValidateMetricsPass
// ============================================================================

void ValidateMetricsPass::run(PipelineModule& module, DiagnosticsEngine& diag) {
    using MetricSet = std::unordered_set<std::string>;

    for (auto& exp : module.experiments) {
        // Build a map: evalVarName -> {metrics...}
        std::unordered_map<std::string, MetricSet> evalMetrics;

        for (const auto& op : exp.ops) {
            if (op.kind != PipelineOpKind::Evaluate) {
                continue;
            }
            const auto& evalOp = op.eval;
            MetricSet ms;
            for (const auto& m : evalOp.metrics) {
                ms.insert(m);
            }
            evalMetrics.emplace(evalOp.varName, std::move(ms));
        }

        if (evalMetrics.empty()) {
            // No evaluation ops; results can't be validated meaningfully.
            continue;
        }

        // Now validate each result field.
        for (const auto& rf : exp.results) {
            auto it = evalMetrics.find(rf.sourceVar);
            if (it == evalMetrics.end()) {
                diag.warn(rf.loc,
                          "result field '" + rf.fieldName +
                              "' refers to unknown evaluation var '" +
                              rf.sourceVar + "'");
                continue;
            }

            const MetricSet& ms = it->second;
            if (ms.find(rf.metricName) == ms.end()) {
                diag.warn(rf.loc,
                          "result field '" + rf.fieldName + "' expects metric '" +
                              rf.metricName + "' from '" + rf.sourceVar +
                              "', but this metric is not produced by that evaluate op");
            }
        }
    }
}

} // namespace syntra