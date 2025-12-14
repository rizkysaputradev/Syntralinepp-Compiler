#pragma once

#include <string>
#include <vector>

#include "syntra/diagnostics.hpp"
#include "syntra/types.hpp"
#include "syntra/ast.hpp"  // for NamedField

namespace syntra {

// ============================================================================
// IR representations of datasets and models
// ============================================================================

struct IRDataset {
    std::string name;        // "mnist"
    DatasetKind kind;        // Image / Tabular / Unknown
    std::string typeName;    // "ImageDataset"
    SourceLocation loc;

    // Non-owning pointers to original fields in the AST.
    // Lifetime is valid as long as the Program stays alive (main.cpp keeps it).
    std::vector<const NamedField*> fields;
};

struct IRModel {
    std::string name;        // "simple_cnn"
    ModelKind kind;          // Torch / Unknown
    std::string typeName;    // "TorchModel"
    SourceLocation loc;

    // Non-owning pointers to original fields in the AST.
    std::vector<const NamedField*> fields;
};

// ============================================================================
// IR representation of experiment pipelines
// ============================================================================

enum class PipelineOpKind {
    Train,
    Evaluate,
};

struct PipelineTrainOp {
    std::string varName;      // "train_run"
    std::string modelName;    // "simple_cnn"
    std::string datasetName;  // "mnist"
    std::string splitName;    // "train"
};

struct PipelineEvaluateOp {
    std::string varName;      // "test_run"
    std::string modelName;    // "simple_cnn"
    std::string datasetName;  // "mnist"
    std::string splitName;    // "test"
    std::vector<std::string> metrics; // ["accuracy", "loss"]
};

struct PipelineOp {
    PipelineOpKind kind;
    SourceLocation loc;

    PipelineTrainOp train;
    PipelineEvaluateOp eval;
};

struct ExperimentResultField {
    std::string fieldName; // "accuracy"
    std::string sourceVar; // "test_run"
    std::string metricName;// "accuracy"
    SourceLocation loc;
};

struct IRExperimentPipeline {
    std::string name;           // "cnn_mnist"
    std::string datasetParam;   // "mnist"
    std::string resultTypeName; // "Metrics"
    SourceLocation loc;

    std::vector<PipelineOp> ops;
    std::vector<ExperimentResultField> results;
};

// ============================================================================
// Whole module IR
// ============================================================================

struct PipelineModule {
    std::vector<IRDataset> datasets;
    std::vector<IRModel> models;
    std::vector<IRExperimentPipeline> experiments;
};

} // namespace syntra
