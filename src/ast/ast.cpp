#include "syntra/ast.hpp"

#include <iomanip>
#include <ostream>
#include <sstream>

namespace syntra {

// Small indent helper
static void printIndent(std::ostream& out, int indent) {
    for (int i = 0; i < indent; ++i) {
        out << ' ';
    }
}

// Forward declarations for recursive dumpers
static void dumpExpr(const ExprPtr& expr, std::ostream& out, int indent);
static void dumpStmt(const StmtPtr& stmt,
                     std::ostream& out,
                     int indent,
                     bool withLocations);

// -----------------------------------------------------------------------------
// Literal / identifier helpers
// -----------------------------------------------------------------------------

static void dumpIdentifier(const IdentifierExpr& id, std::ostream& out) {
    out << id.name;
}

static void dumpIntegerLiteral(const IntegerLiteralExpr& lit, std::ostream& out) {
    out << lit.value;
}

static void dumpFloatLiteral(const FloatLiteralExpr& lit, std::ostream& out) {
    out << lit.value;
}

static void dumpStringLiteral(const StringLiteralExpr& lit, std::ostream& out) {
    out << '"' << lit.value << '"';
}

static void dumpBoolLiteral(const BoolLiteralExpr& lit, std::ostream& out) {
    out << (lit.value ? "true" : "false");
}

// -----------------------------------------------------------------------------
// Expressions
// -----------------------------------------------------------------------------

static void dumpTupleExpr(const TupleExpr& tup,
                          std::ostream& out,
                          int indent) {
    (void)indent;
    out << "(";
    for (std::size_t i = 0; i < tup.elements.size(); ++i) {
        if (i > 0) out << ", ";
        dumpExpr(tup.elements[i], out, 0);
    }
    if (tup.elements.size() == 1) {
        out << ","; // distinguish (x) vs (x,)
    }
    out << ")";
}

static void dumpListExpr(const ListExpr& list,
                         std::ostream& out,
                         int indent) {
    (void)indent;
    out << "[";
    for (std::size_t i = 0; i < list.elements.size(); ++i) {
        if (i > 0) out << ", ";
        dumpExpr(list.elements[i], out, 0);
    }
    out << "]";
}

static void dumpMemberAccessExpr(const MemberAccessExpr& m,
                                 std::ostream& out,
                                 int indent) {
    (void)indent;
    dumpExpr(m.base, out, 0);
    out << "." << m.member;
}

static void dumpCallExpr(const CallExpr& call,
                         std::ostream& out,
                         int indent) {
    (void)indent;
    dumpExpr(call.callee, out, 0);
    out << "(";
    for (std::size_t i = 0; i < call.arguments.size(); ++i) {
        if (i > 0) out << ", ";
        dumpExpr(call.arguments[i], out, 0);
    }
    out << ")";
}

static void dumpWithConfigExpr(const WithConfigExpr& wc,
                               std::ostream& out,
                               int indent) {
    (void)indent;
    dumpExpr(wc.base, out, 0);
    out << " with { ";
    for (std::size_t i = 0; i < wc.fields.size(); ++i) {
        const auto& nf = wc.fields[i];
        if (i > 0) out << ", ";
        out << nf.name << " = ";
        dumpExpr(nf.value, out, 0);
    }
    out << " }";
}

static void dumpExpr(const ExprPtr& expr,
                     std::ostream& out,
                     int indent) {
    if (!expr) {
        out << "<null-expr>";
        return;
    }

    switch (expr->kind) {
    case ExprKind::Identifier:
        dumpIdentifier(static_cast<const IdentifierExpr&>(*expr), out);
        break;
    case ExprKind::IntegerLiteral:
        dumpIntegerLiteral(static_cast<const IntegerLiteralExpr&>(*expr), out);
        break;
    case ExprKind::FloatLiteral:
        dumpFloatLiteral(static_cast<const FloatLiteralExpr&>(*expr), out);
        break;
    case ExprKind::StringLiteral:
        dumpStringLiteral(static_cast<const StringLiteralExpr&>(*expr), out);
        break;
    case ExprKind::BoolLiteral:
        dumpBoolLiteral(static_cast<const BoolLiteralExpr&>(*expr), out);
        break;
    case ExprKind::Tuple:
        dumpTupleExpr(static_cast<const TupleExpr&>(*expr), out, indent);
        break;
    case ExprKind::List:
        dumpListExpr(static_cast<const ListExpr&>(*expr), out, indent);
        break;
    case ExprKind::MemberAccess:
        dumpMemberAccessExpr(static_cast<const MemberAccessExpr&>(*expr), out, indent);
        break;
    case ExprKind::Call:
        dumpCallExpr(static_cast<const CallExpr&>(*expr), out, indent);
        break;
    case ExprKind::WithConfig:
        dumpWithConfigExpr(static_cast<const WithConfigExpr&>(*expr), out, indent);
        break;
    default:
        out << "<expr kind=" << static_cast<int>(expr->kind) << ">";
        break;
    }
}

// -----------------------------------------------------------------------------
// Named fields (used in configs and returns)
// -----------------------------------------------------------------------------

static void dumpNamedFields(const std::vector<NamedField>& fields,
                            std::ostream& out,
                            int indent) {
    for (const auto& nf : fields) {
        printIndent(out, indent);
        out << nf.name << " = ";
        dumpExpr(nf.value, out, indent);
        out << "\n";
    }
}

// -----------------------------------------------------------------------------
// Statements
// -----------------------------------------------------------------------------

static void dumpLetStmt(const LetStmt& st,
                        std::ostream& out,
                        int indent,
                        bool withLocations) {
    printIndent(out, indent);
    out << "let " << st.name << " = ";
    dumpExpr(st.expr, out, indent);
    if (withLocations) {
        out << "    # at " << st.loc.line << ":" << st.loc.column;
    }
    out << "\n";
}

static void dumpReturnStmt(const ReturnStmt& st,
                           std::ostream& out,
                           int indent,
                           bool withLocations) {
    printIndent(out, indent);
    out << "return {\n";
    dumpNamedFields(st.fields, out, indent + 4);
    printIndent(out, indent);
    out << "}";
    if (withLocations) {
        out << "    # at " << st.loc.line << ":" << st.loc.column;
    }
    out << "\n";
}

static void dumpStmt(const StmtPtr& stmt,
                     std::ostream& out,
                     int indent,
                     bool withLocations) {
    if (!stmt) {
        printIndent(out, indent);
        out << "<null-stmt>\n";
        return;
    }

    switch (stmt->kind) {
    case StmtKind::Let:
        dumpLetStmt(static_cast<const LetStmt&>(*stmt), out, indent, withLocations);
        break;
    case StmtKind::Return:
        dumpReturnStmt(static_cast<const ReturnStmt&>(*stmt), out, indent, withLocations);
        break;
    default:
        printIndent(out, indent);
        out << "<stmt kind=" << static_cast<int>(stmt->kind) << ">\n";
        break;
    }
}

// -----------------------------------------------------------------------------
// Declarations (datasets, models, experiments)
// -----------------------------------------------------------------------------

static void dumpDatasetDecl(const DatasetDecl& d,
                            std::ostream& out,
                            bool withLocations) {
    out << "dataset " << d.name << " : " << d.datasetType;
    if (withLocations) {
        out << "    # at " << d.loc.line << ":" << d.loc.column;
    }
    out << "\n{\n";
    dumpNamedFields(d.fields, out, /*indent=*/4);
    out << "}\n\n";
}

static void dumpModelDecl(const ModelDecl& m,
                          std::ostream& out,
                          bool withLocations) {
    out << "model " << m.name << " : " << m.modelType;
    if (withLocations) {
        out << "    # at " << m.loc.line << ":" << m.loc.column;
    }
    out << "\n{\n";
    dumpNamedFields(m.fields, out, /*indent=*/4);
    out << "}\n\n";
}

static void dumpExperimentDecl(const ExperimentDecl& e,
                               std::ostream& out,
                               bool withLocations) {
    out << "experiment " << e.name << "(" << e.datasetParam << ") -> "
        << e.resultType;
    if (withLocations) {
        out << "    # at " << e.loc.line << ":" << e.loc.column;
    }
    out << "\n{\n";

    for (const auto& st : e.body) {
        dumpStmt(st, out, /*indent=*/4, withLocations);
    }

    out << "}\n\n";
}

// -----------------------------------------------------------------------------
// Program
// -----------------------------------------------------------------------------

void dumpAST(const Program& program,
             std::ostream& out,
             bool withLocations) {
    out << "# === SyntraLine++ AST dump ===\n";

    if (!program.datasets.empty()) {
        out << "\n# Datasets\n";
        for (const auto& d : program.datasets) {
            dumpDatasetDecl(d, out, withLocations);
        }
    }

    if (!program.models.empty()) {
        out << "\n# Models\n";
        for (const auto& m : program.models) {
            dumpModelDecl(m, out, withLocations);
        }
    }

    if (!program.experiments.empty()) {
        out << "\n# Experiments\n";
        for (const auto& e : program.experiments) {
            dumpExperimentDecl(e, out, withLocations);
        }
    }

    out << "# === End AST dump ===\n";
}

std::string astToString(const Program& program,
                        bool withLocations) {
    std::ostringstream oss;
    dumpAST(program, oss, withLocations);
    return oss.str();
}

} // namespace syntra