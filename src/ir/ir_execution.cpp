#include "syntra/ir_execution.hpp"

#include <cmath>
#include <random>

namespace syntra {

namespace {

/// Very simple deterministic RNG seeded from a string,
/// so runs are stable for a given experiment name.
std::mt19937_64 makeRngFromName(const std::string& name) {
    std::uint64_t seed = 0xcbf29ce484222325ULL; // FNV-ish
    for (char c : name) {
        seed ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        seed *= 0x100000001b3ULL;
    }
    return std::mt19937_64(seed);
}

} // namespace

// ============================================================================
// Public entrypoints
// ============================================================================

ExperimentRunResult PipelineExecutor::runFirstExperiment(const PipelineModule& module) {
    ExperimentRunResult result;
    if (module.experiments.empty()) {
        diag_.error(SourceLocation{}, "no experiments defined in module");
        return result;
    }
    return runExperiment(module.experiments.front());
}

ExperimentRunResult PipelineExecutor::runExperimentByName(const PipelineModule& module,
                                                          const std::string& name) {
    for (const auto& e : module.experiments) {
        if (e.name == name) {
            return runExperiment(e);
        }
    }
    diag_.error(SourceLocation{}, "no experiment named '" + name + "' found in module");
    return ExperimentRunResult{};
}

// ============================================================================
// Internal – experiment run
// ============================================================================

ExperimentRunResult PipelineExecutor::runExperiment(const IRExperimentPipeline& exp) {
    ExperimentRunResult result;
    result.experimentName = exp.name;

    // For simulation, we keep a simple map: evalVarName -> metrics
    std::unordered_map<std::string, MetricMap> evalResults;

    // Seed RNG based on experiment name so results are stable
    std::mt19937_64 rng = makeRngFromName(exp.name);

    for (const auto& op : exp.ops) {
        switch (op.kind) {
        case PipelineOpKind::Train: {
            MetricMap trainStats = simulateTrain(op.train);
            // For now we don't store train metrics; could be extended later.
            (void)trainStats;
            break;
        }
        case PipelineOpKind::Evaluate: {
            MetricMap evalStats = simulateEvaluate(op.eval);

            // Adjust randomness with RNG so runs differ between experiments.
            // We'll just jitter values slightly.
            for (auto& kv : evalStats) {
                double noise = std::uniform_real_distribution<double>(
                                   -0.02, 0.02)(rng);
                kv.second = std::max(0.0, std::min(1.0, kv.second + noise));
            }

            evalResults[op.eval.varName] = std::move(evalStats);
            break;
        }
        }
    }

    // Wire through the explicit or canonicalized return mapping.
    for (const auto& rf : exp.results) {
        auto it = evalResults.find(rf.sourceVar);
        if (it == evalResults.end()) {
            diag_.warn(rf.loc,
                       "simulate: no evaluation results for '" +
                           rf.sourceVar + "'; field '" + rf.fieldName +
                           "' will be set to 0.0");
            result.metrics[rf.fieldName] = 0.0;
            continue;
        }

        const MetricMap& stats = it->second;
        auto mit = stats.find(rf.metricName);
        if (mit == stats.end()) {
            diag_.warn(rf.loc,
                       "simulate: evaluation results for '" + rf.sourceVar +
                           "' do not contain metric '" + rf.metricName +
                           "'; field '" + rf.fieldName +
                           "' will be set to 0.0");
            result.metrics[rf.fieldName] = 0.0;
            continue;
        }

        result.metrics[rf.fieldName] = mit->second;
    }

    return result;
}

// ============================================================================
// Internal – op simulation
// ============================================================================

MetricMap PipelineExecutor::simulateTrain(const PipelineTrainOp& trainOp) {
    MetricMap stats;

    // Very simple fake training: loss decreases from ~2.3 to ~0.5
    // depending on the model/dataset names in a deterministic way.
    std::string key = trainOp.modelName + "::" + trainOp.datasetName + "::" + trainOp.splitName;
    std::mt19937_64 rng = makeRngFromName(key);

    double baseLoss = 2.3;
    double improvement = std::uniform_real_distribution<double>(0.5, 1.5)(rng);
    double finalLoss = std::max(0.0, baseLoss - improvement);

    stats["loss"] = finalLoss;
    return stats;
}

MetricMap PipelineExecutor::simulateEvaluate(const PipelineEvaluateOp& evalOp) {
    MetricMap stats;

    // Very simple fake evaluation:
    //  - loss in [0.2, 2.5]
    //  - accuracy in [0.1, 0.99] if requested
    std::string key = evalOp.modelName + "::" + evalOp.datasetName + "::" + evalOp.splitName;
    std::mt19937_64 rng = makeRngFromName(key);

    double loss = std::uniform_real_distribution<double>(0.15, 2.5)(rng);
    stats["loss"] = loss;

    // Only produce accuracy if the pipeline requested it
    bool wantsAccuracy = false;
    for (const auto& m : evalOp.metrics) {
        if (m == "accuracy") {
            wantsAccuracy = true;
            break;
        }
    }
    if (wantsAccuracy) {
        double accuracy = std::uniform_real_distribution<double>(0.1, 0.99)(rng);
        stats["accuracy"] = accuracy;
    }

    // Other metrics could be added here in the future.
    return stats;
}

} // namespace syntra