#pragma once

#include <string>

namespace syntra {

struct PipelineModule;

/// Serialize a PipelineModule (datasets, models, experiments, ops, results)
/// into a JSON string.
///
/// The JSON schema is intentionally simple and stable:
///
/// {
///   "datasets": [
///     { "name": "...", "type": "..." },
///     ...
///   ],
///   "models": [
///     { "name": "...", "type": "..." },
///     ...
///   ],
///   "experiments": [
///     {
///       "name": "...",
///       "dataset_param": "...",
///       "result_type": "...",
///       "ops": [
///         {
///           "kind": "train",
///           "var": "...",
///           "model": "...",
///           "dataset": "...",
///           "split": "..."
///         },
///         {
///           "kind": "evaluate",
///           "var": "...",
///           "model": "...",
///           "dataset": "...",
///           "split": "...",
///           "metrics": ["accuracy", "loss"]
///         }
///       ],
///       "results": [
///         {
///           "field": "accuracy",
///           "source_var": "test_run",
///           "metric": "accuracy"
///         },
///         ...
///       ]
///     },
///     ...
///   ]
/// }
///
/// For now we omit detailed dataset/model field configs from JSON.
/// We can extend this later once the runtime is ready to consume them.
std::string pipelineToJson(const PipelineModule &module, bool pretty = true);

} // namespace syntra