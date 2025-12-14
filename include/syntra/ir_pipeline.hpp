#pragma once

#include <string>
#include <unordered_map>

#include "syntra/ast.hpp"
#include "syntra/ir.hpp"
#include "syntra/diagnostics.hpp"
#include "syntra/types.hpp"

namespace syntra {

/// Builder that lowers a semantically-valid Program into a PipelineModule.
class PipelineBuilder {
public:
    explicit PipelineBuilder(DiagnosticsEngine& diag)
        : diag_(diag) {}

    PipelineModule build(const Program& program);

private:
    // Helpers for building IR from AST
    void buildDatasets(const Program& program, PipelineModule& module);
    void buildModels(const Program& program, PipelineModule& module);
    void buildExperiments(const Program& program, PipelineModule& module);

    IRExperimentPipeline buildExperimentPipeline(const ExperimentDecl& decl);

    PipelineOp lowerLetStatement(const ExperimentDecl& decl, const LetStmt& letStmt);
    void lowerReturnStatement(const ExperimentDecl& decl,
                              const ReturnStmt& retStmt,
                              IRExperimentPipeline& pipeline);

    // Expression pattern helpers
    // We only support specific shapes in v0.1:
    //   train(simple_cnn, mnist.train)
    //   evaluate(simple_cnn, mnist.test) with { metrics = [accuracy, loss] }
    bool matchTrainCall(const Expr& expr,
                        std::string& outModelName,
                        std::string& outDatasetName,
                        std::string& outSplitName);

    bool matchEvaluateCallWithMetrics(const Expr& expr,
                                      std::string& outModelName,
                                      std::string& outDatasetName,
                                      std::string& outSplitName,
                                      std::vector<std::string>& outMetrics);

    bool matchMemberAccess(const Expr& expr,
                           std::string& outBaseName,
                           std::string& outMemberName);

    // Warn on unknown shapes but keep going so IR is still somewhat usable.
    void warnUnsupportedLetPattern(const LetStmt& letStmt);
    void warnUnsupportedReturnField(const NamedField& field);

    DiagnosticsEngine& diag_;
};

} // namespace syntra
