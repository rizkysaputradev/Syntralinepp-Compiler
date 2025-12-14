#include "syntra/sema.hpp"

#include <sstream>

namespace syntra {

void SemanticAnalyzer::analyze(const Program& program) {
    // Pass 1: register all top-level declarations.
    registerDatasets(program);
    registerModels(program);
    registerExperiments(program);

    // Pass 2: check each declaration in more detail.
    for (const auto& d : program.datasets) {
        checkDatasetDecl(d);
    }
    for (const auto& m : program.models) {
        checkModelDecl(m);
    }
    for (const auto& e : program.experiments) {
        checkExperimentDecl(e);
    }
}

// ============================================================================
// Registration
// ============================================================================

void SemanticAnalyzer::registerDatasets(const Program& program) {
    for (const auto& d : program.datasets) {
        symbols_.insert(SymbolKind::Dataset,
                        d.name,
                        d.datasetType,
                        d.loc,
                        diag_);
    }
}

void SemanticAnalyzer::registerModels(const Program& program) {
    for (const auto& m : program.models) {
        symbols_.insert(SymbolKind::Model,
                        m.name,
                        m.modelType,
                        m.loc,
                        diag_);
    }
}

void SemanticAnalyzer::registerExperiments(const Program& program) {
    for (const auto& e : program.experiments) {
        symbols_.insert(SymbolKind::Experiment,
                        e.name,
                        e.resultType,
                        e.loc,
                        diag_);
    }
}

// ============================================================================
// Checks
// ============================================================================

void SemanticAnalyzer::checkDatasetDecl(const DatasetDecl& decl) {
    DatasetKind dk = datasetKindFromTypeName(decl.datasetType);
    if (dk == DatasetKind::Unknown) {
        std::ostringstream oss;
        oss << "unknown dataset type '" << decl.datasetType
            << "' (expected e.g. ImageDataset or TabularDataset)";
        diag_.error(decl.loc, oss.str());
    }

    // In v0.1 we won't deeply validate fields; later we can:
    //  - check for required fields like source/shape/batch/etc.
    //  - ensure field types match expectations.
}

void SemanticAnalyzer::checkModelDecl(const ModelDecl& decl) {
    ModelKind mk = modelKindFromTypeName(decl.modelType);
    if (mk == ModelKind::Unknown) {
        std::ostringstream oss;
        oss << "unknown model type '" << decl.modelType
            << "' (expected e.g. TorchModel)";
        diag_.error(decl.loc, oss.str());
    }

    // Later:
    //  - check required fields like arch, lr, epochs, optimizer.
}

void SemanticAnalyzer::checkExperimentDecl(const ExperimentDecl& decl) {
    // Check result type
    ResultKind rk = resultKindFromTypeName(decl.resultType);
    if (rk == ResultKind::Unknown) {
        std::ostringstream oss;
        oss << "unknown experiment result type '" << decl.resultType
            << "' (expected e.g. Metrics)";
        diag_.error(decl.loc, oss.str());
    }

    // Check that the dataset parameter refers to a known dataset symbol.
    const Symbol* sym = symbols_.lookup(decl.datasetParam);
    if (!sym) {
        std::ostringstream oss;
        oss << "unknown dataset '" << decl.datasetParam
            << "' in experiment parameter";
        diag_.error(decl.loc, oss.str());
    } else if (sym->kind != SymbolKind::Dataset) {
        std::ostringstream oss;
        oss << "experiment parameter '" << decl.datasetParam
            << "' does not name a dataset (found " << decl.datasetParam << " as non-dataset symbol)";
        diag_.error(decl.loc, oss.str());
    }

    // Later:
    //  - analyze statements in decl.body:
    //      * resolve identifiers to symbols,
    //      * ensure train/evaluate are used correctly,
    //      * check metrics exist and are valid for the task.
}

} // namespace syntra
