#include "syntra/ir_pipeline.hpp"

#include <sstream>

namespace syntra {

PipelineModule PipelineBuilder::build(const Program& program) {
    PipelineModule module;

    buildDatasets(program, module);
    buildModels(program, module);
    buildExperiments(program, module);

    return module;
}

// ============================================================================
// Build datasets / models
// ============================================================================

void PipelineBuilder::buildDatasets(const Program& program, PipelineModule& module) {
    for (const auto& d : program.datasets) {
        IRDataset ir;
        ir.name     = d.name;
        ir.typeName = d.datasetType;
        ir.kind     = datasetKindFromTypeName(d.datasetType);
        ir.loc      = d.loc;

        ir.fields.clear();
        ir.fields.reserve(d.fields.size());
        for (const auto& field : d.fields) {
            ir.fields.push_back(&field); // store pointer, no copy
        }

        module.datasets.push_back(std::move(ir));
    }
}

void PipelineBuilder::buildModels(const Program& program, PipelineModule& module) {
    for (const auto& m : program.models) {
        IRModel ir;
        ir.name     = m.name;
        ir.typeName = m.modelType;
        ir.kind     = modelKindFromTypeName(m.modelType);
        ir.loc      = m.loc;

        ir.fields.clear();
        ir.fields.reserve(m.fields.size());
        for (const auto& field : m.fields) {
            ir.fields.push_back(&field); // store pointer, no copy
        }

        module.models.push_back(std::move(ir));
    }
}

// ============================================================================
// Build experiments
// ============================================================================

void PipelineBuilder::buildExperiments(const Program& program, PipelineModule& module) {
    for (const auto& e : program.experiments) {
        IRExperimentPipeline pipe = buildExperimentPipeline(e);
        module.experiments.push_back(std::move(pipe));
    }
}

IRExperimentPipeline PipelineBuilder::buildExperimentPipeline(const ExperimentDecl& decl) {
    IRExperimentPipeline pipe;
    pipe.name           = decl.name;
    pipe.datasetParam   = decl.datasetParam;
    pipe.resultTypeName = decl.resultType;
    pipe.loc            = decl.loc;

    for (const auto& stmtPtr : decl.body) {
        if (!stmtPtr) continue;

        switch (stmtPtr->kind) {
        case StmtKind::Let: {
            const LetStmt* letStmt = static_cast<const LetStmt*>(stmtPtr.get());
            PipelineOp op = lowerLetStatement(decl, *letStmt);
            pipe.ops.push_back(std::move(op));
            break;
        }
        case StmtKind::Return: {
            const ReturnStmt* retStmt = static_cast<const ReturnStmt*>(stmtPtr.get());
            lowerReturnStatement(decl, *retStmt, pipe);
            break;
        }
        }
    }

    return pipe;
}

// ============================================================================
// Lower let statements
// ============================================================================

PipelineOp PipelineBuilder::lowerLetStatement(const ExperimentDecl& decl,
                                              const LetStmt& letStmt) {
    PipelineOp op;
    op.loc = letStmt.loc;

    std::string modelName;
    std::string datasetName;
    std::string splitName;
    std::vector<std::string> metrics;

    // We only support two shapes in v0.1:
    //   let x = train(model, ds.split)
    //   let x = evaluate(model, ds.split) with { metrics = [...] }

    if (matchTrainCall(*letStmt.expr, modelName, datasetName, splitName)) {
        op.kind = PipelineOpKind::Train;
        op.train.varName     = letStmt.name;
        op.train.modelName   = modelName;
        op.train.datasetName = datasetName;
        op.train.splitName   = splitName;
    } else if (matchEvaluateCallWithMetrics(*letStmt.expr,
                                            modelName,
                                            datasetName,
                                            splitName,
                                            metrics)) {
        op.kind = PipelineOpKind::Evaluate;
        op.eval.varName     = letStmt.name;
        op.eval.modelName   = modelName;
        op.eval.datasetName = datasetName;
        op.eval.splitName   = splitName;
        op.eval.metrics     = std::move(metrics);
    } else {
        warnUnsupportedLetPattern(letStmt);
        // Create a dummy op to keep pipeline shape consistent.
        op.kind = PipelineOpKind::Train;
        op.train.varName     = letStmt.name;
        op.train.modelName   = "<unknown>";
        op.train.datasetName = decl.datasetParam;
        op.train.splitName   = "<unknown>";
    }

    return op;
}

// ============================================================================
// Lower return statement
// ============================================================================

void PipelineBuilder::lowerReturnStatement(const ExperimentDecl& decl,
                                           const ReturnStmt& retStmt,
                                           IRExperimentPipeline& pipeline) {
    (void)decl;
    for (const auto& field : retStmt.fields) {
        std::string baseName;
        std::string metricName;
        if (!matchMemberAccess(*field.value, baseName, metricName)) {
            warnUnsupportedReturnField(field);
            continue;
        }

        ExperimentResultField rf;
        rf.fieldName  = field.name;
        rf.sourceVar  = baseName;
        rf.metricName = metricName;
        rf.loc        = field.loc;
        pipeline.results.push_back(std::move(rf));
    }
}

// ============================================================================
// Expression pattern helpers
// ============================================================================

// train(simple_cnn, mnist.train)
bool PipelineBuilder::matchTrainCall(const Expr& expr,
                                     std::string& outModelName,
                                     std::string& outDatasetName,
                                     std::string& outSplitName) {
    const Expr* e = &expr;

    // Must be CallExpr
    if (e->kind != ExprKind::Call) {
        return false;
    }
    const auto& call = static_cast<const CallExpr&>(*e);

    // callee must be Identifier("train")
    if (call.callee->kind != ExprKind::Identifier) {
        return false;
    }
    const auto& calleeId = static_cast<const IdentifierExpr&>(*call.callee);
    if (calleeId.name != "train") {
        return false;
    }

    if (call.arguments.size() != 2) {
        return false;
    }

    // First arg: model name
    if (call.arguments[0]->kind != ExprKind::Identifier) {
        return false;
    }
    const auto& modelId = static_cast<const IdentifierExpr&>(*call.arguments[0]);
    outModelName = modelId.name;

    // Second arg: mnist.train (MemberAccessExpr)
    std::string dsName;
    std::string split;
    if (!matchMemberAccess(*call.arguments[1], dsName, split)) {
        return false;
    }
    outDatasetName = dsName;
    outSplitName   = split;

    return true;
}

// evaluate(simple_cnn, mnist.test) with { metrics = [accuracy, loss] }
bool PipelineBuilder::matchEvaluateCallWithMetrics(const Expr& expr,
                                                   std::string& outModelName,
                                                   std::string& outDatasetName,
                                                   std::string& outSplitName,
                                                   std::vector<std::string>& outMetrics) {
    const Expr* e = &expr;

    // Outer must be WithConfigExpr
    if (e->kind != ExprKind::WithConfig) {
        return false;
    }

    const auto& withExpr = static_cast<const WithConfigExpr&>(*e);
    const Expr& base = *withExpr.base;

    // Base must be CallExpr
    if (base.kind != ExprKind::Call) {
        return false;
    }
    const auto& call = static_cast<const CallExpr&>(base);

    // callee must be Identifier("evaluate")
    if (call.callee->kind != ExprKind::Identifier) {
        return false;
    }
    const auto& calleeId = static_cast<const IdentifierExpr&>(*call.callee);
    if (calleeId.name != "evaluate") {
        return false;
    }

    if (call.arguments.size() != 2) {
        return false;
    }

    // First arg: model name
    if (call.arguments[0]->kind != ExprKind::Identifier) {
        return false;
    }
    const auto& modelId = static_cast<const IdentifierExpr&>(*call.arguments[0]);
    outModelName = modelId.name;

    // Second arg: mnist.test (MemberAccessExpr)
    std::string dsName;
    std::string split;
    if (!matchMemberAccess(*call.arguments[1], dsName, split)) {
        return false;
    }
    outDatasetName = dsName;
    outSplitName   = split;

    // Now parse config fields: look for "metrics = [ ... ]"
    for (const auto& field : withExpr.fields) {
        if (field.name == "metrics") {
            if (field.value->kind != ExprKind::List) {
                diag_.error(field.loc, "expected list literal for 'metrics' field");
                return false;
            }
            const auto& list = static_cast<const ListExpr&>(*field.value);
            outMetrics.clear();
            for (const auto& elem : list.elements) {
                if (elem->kind != ExprKind::Identifier) {
                    diag_.error(elem->loc, "expected identifier for metric name");
                    continue;
                }
                const auto& metricId = static_cast<const IdentifierExpr&>(*elem);
                outMetrics.push_back(metricId.name);
            }
            return true;
        }
    }

    // No metrics field found
    diag_.error(withExpr.loc, "missing 'metrics' field in evaluate(...) with { ... }");
    return false;
}

// mnist.train OR test_run.accuracy
bool PipelineBuilder::matchMemberAccess(const Expr& expr,
                                        std::string& outBaseName,
                                        std::string& outMemberName) {
    if (expr.kind != ExprKind::MemberAccess) {
        return false;
    }
    const auto& mem = static_cast<const MemberAccessExpr&>(expr);
    if (mem.base->kind != ExprKind::Identifier) {
        return false;
    }
    const auto& baseId = static_cast<const IdentifierExpr&>(*mem.base);

    outBaseName   = baseId.name;
    outMemberName = mem.member;
    return true;
}

// ============================================================================
// Warnings for unsupported patterns
// ============================================================================

void PipelineBuilder::warnUnsupportedLetPattern(const LetStmt& letStmt) {
    std::ostringstream oss;
    oss << "unsupported let-pattern for lowering into pipeline IR; "
        << "expected train(model, dataset.split) or evaluate(model, dataset.split) with { metrics = [...] }";
    diag_.warn(letStmt.loc, oss.str());
}

void PipelineBuilder::warnUnsupportedReturnField(const NamedField& field) {
    std::ostringstream oss;
    oss << "unsupported return field expression for '" << field.name
        << "'; expected something like test_run.accuracy";
    diag_.warn(field.loc, oss.str());
}

} // namespace syntra
