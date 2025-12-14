#pragma once

#include "syntra/ast.hpp"
#include "syntra/symbol_table.hpp"
#include "syntra/diagnostics.hpp"
#include "syntra/types.hpp"

namespace syntra {

/// Semantic analysis for SyntraLine++ v0.1.
///
/// Responsibilities:
///  - build a global symbol table for datasets, models, experiments.
///  - check for redefinitions.
///  - validate basic type names (dataset/model/experiment result).
///  - check that experiments refer to existing datasets.
///
/// For v0.1 we *do not* yet perform full expression-level type checking;
/// that can be added incrementally on top of this skeleton.
class SemanticAnalyzer {
public:
    explicit SemanticAnalyzer(DiagnosticsEngine& diag)
        : diag_(diag) {}

    /// Perform semantic analysis on the given Program.
    /// This may modify the Program in-place in future versions (e.g. to attach
    /// resolved type info). For now it only reads from it.
    void analyze(const Program& program);

private:
    void registerDatasets(const Program& program);
    void registerModels(const Program& program);
    void registerExperiments(const Program& program);

    void checkDatasetDecl(const DatasetDecl& decl);
    void checkModelDecl(const ModelDecl& decl);
    void checkExperimentDecl(const ExperimentDecl& decl);

    DiagnosticsEngine& diag_;
    SymbolTable symbols_;
};

} // namespace syntra
