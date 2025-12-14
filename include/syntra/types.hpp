#pragma once

#include <string>
#include <optional>

namespace syntra {

/// High-level kinds of datasets recognized by v0.1.
enum class DatasetKind {
    Image,
    Tabular,
    Unknown,
};

/// High-level kinds of models recognized by v0.1.
enum class ModelKind {
    Torch,
    Unknown,
};

/// Result type of an experiment.
enum class ResultKind {
    Metrics,
    Unknown,
};

inline DatasetKind datasetKindFromTypeName(const std::string& typeName) {
    if (typeName == "ImageDataset") {
        return DatasetKind::Image;
    }
    if (typeName == "TabularDataset") {
        return DatasetKind::Tabular;
    }
    return DatasetKind::Unknown;
}

inline ModelKind modelKindFromTypeName(const std::string& typeName) {
    if (typeName == "TorchModel") {
        return ModelKind::Torch;
    }
    return ModelKind::Unknown;
}

inline ResultKind resultKindFromTypeName(const std::string& typeName) {
    if (typeName == "Metrics") {
        return ResultKind::Metrics;
    }
    return ResultKind::Unknown;
}

} // namespace syntra
