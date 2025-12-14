#pragma once

#include <memory>
#include <string>
#include <vector>
#include <ostream>

#include "syntra/diagnostics.hpp"

namespace syntra {

// Forward declarations
struct Expr;
struct Stmt;

using ExprPtr  = std::unique_ptr<Expr>;
using StmtPtr  = std::unique_ptr<Stmt>;

// ============================================================================
// Expressions
// ============================================================================

enum class ExprKind {
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BoolLiteral,
    Tuple,
    List,
    MemberAccess,
    Call,
    WithConfig, // e.g. evaluate(...) with { metrics = [...] }
};

struct Expr {
    ExprKind kind;
    SourceLocation loc;

    explicit Expr(ExprKind k, const SourceLocation& l)
        : kind(k), loc(l) {}

    virtual ~Expr() = default;
};

// IdentifierExpr: e.g. mnist, simple_cnn, accuracy
struct IdentifierExpr : public Expr {
    std::string name;

    IdentifierExpr(const std::string& n, const SourceLocation& l)
        : Expr(ExprKind::Identifier, l), name(n) {}
};

// Literals
struct IntegerLiteralExpr : public Expr {
    std::string value; // keep as string for now (we can parse later)

    IntegerLiteralExpr(const std::string& v, const SourceLocation& l)
        : Expr(ExprKind::IntegerLiteral, l), value(v) {}
};

struct FloatLiteralExpr : public Expr {
    std::string value;

    FloatLiteralExpr(const std::string& v, const SourceLocation& l)
        : Expr(ExprKind::FloatLiteral, l), value(v) {}
};

struct StringLiteralExpr : public Expr {
    std::string value;

    StringLiteralExpr(const std::string& v, const SourceLocation& l)
        : Expr(ExprKind::StringLiteral, l), value(v) {}
};

struct BoolLiteralExpr : public Expr {
    bool value;

    BoolLiteralExpr(bool v, const SourceLocation& l)
        : Expr(ExprKind::BoolLiteral, l), value(v) {}
};

// TupleExpr: e.g. (28, 28)
struct TupleExpr : public Expr {
    std::vector<ExprPtr> elements;

    TupleExpr(std::vector<ExprPtr> elems, const SourceLocation& l)
        : Expr(ExprKind::Tuple, l), elements(std::move(elems)) {}
};

// ListExpr: e.g. [accuracy, loss]
struct ListExpr : public Expr {
    std::vector<ExprPtr> elements;

    ListExpr(std::vector<ExprPtr> elems, const SourceLocation& l)
        : Expr(ExprKind::List, l), elements(std::move(elems)) {}
};

// MemberAccessExpr: e.g. mnist.train or test_run.accuracy
struct MemberAccessExpr : public Expr {
    ExprPtr base;
    std::string member;

    MemberAccessExpr(ExprPtr b, const std::string& m, const SourceLocation& l)
        : Expr(ExprKind::MemberAccess, l),
          base(std::move(b)),
          member(m) {}
};

// CallExpr: e.g. train(simple_cnn, mnist.train)
struct CallExpr : public Expr {
    ExprPtr callee;
    std::vector<ExprPtr> arguments;

    CallExpr(ExprPtr c, std::vector<ExprPtr> args, const SourceLocation& l)
        : Expr(ExprKind::Call, l),
          callee(std::move(c)),
          arguments(std::move(args)) {}
};

// Field assignment used in config blocks and return blocks:
//   name = expr
struct NamedField {
    std::string name;
    ExprPtr value;
    SourceLocation loc;
};

// WithConfigExpr: e.g. evaluate(...) with { metrics = [accuracy, loss] }
struct WithConfigExpr : public Expr {
    ExprPtr base;
    std::vector<NamedField> fields;

    WithConfigExpr(ExprPtr b,
                   std::vector<NamedField> f,
                   const SourceLocation& l)
        : Expr(ExprKind::WithConfig, l),
          base(std::move(b)),
          fields(std::move(f)) {}
};

// ============================================================================
// Statements (used inside experiment bodies)
// ============================================================================

enum class StmtKind {
    Let,
    Return,
};

struct Stmt {
    StmtKind kind;
    SourceLocation loc;

    Stmt(StmtKind k, const SourceLocation& l)
        : kind(k), loc(l) {}

    virtual ~Stmt() = default;
};

// let <name> = <expr>
struct LetStmt : public Stmt {
    std::string name;
    ExprPtr expr;

    LetStmt(const std::string& n, ExprPtr e, const SourceLocation& l)
        : Stmt(StmtKind::Let, l),
          name(n),
          expr(std::move(e)) {}
};

// return { field = expr, ... }
struct ReturnStmt : public Stmt {
    std::vector<NamedField> fields;

    explicit ReturnStmt(std::vector<NamedField> f, const SourceLocation& l)
        : Stmt(StmtKind::Return, l),
          fields(std::move(f)) {}
};

// ============================================================================
// Top-level declarations
// ============================================================================

struct DatasetDecl {
    std::string name;       // e.g. "mnist"
    std::string datasetType;// e.g. "ImageDataset"
    std::vector<NamedField> fields;
    SourceLocation loc;
};

struct ModelDecl {
    std::string name;   // e.g. "simple_cnn"
    std::string modelType; // e.g. "TorchModel"
    std::vector<NamedField> fields;
    SourceLocation loc;
};

struct ExperimentDecl {
    std::string name;          // cnn_mnist
    std::string datasetParam;  // mnist
    std::string resultType;    // Metrics
    std::vector<StmtPtr> body; // let ...; let ...; return {...}
    SourceLocation loc;
};

// Entire program: a collection of top-level declarations
struct Program {
    std::vector<DatasetDecl> datasets;
    std::vector<ModelDecl> models;
    std::vector<ExperimentDecl> experiments;
};

// -----------------------------------------------------------------------------
// AST dump / pretty-print helpers
// -----------------------------------------------------------------------------
/// Dump a human-readable representation of the AST to the given stream.
/// This is intended for debugging and language development, not as a stable
/// serialization format.
void dumpAST(const Program& program,
             std::ostream& out,
             bool withLocations = true);

/// Convenience helper: return the AST dump as a string.
std::string astToString(const Program& program,
                        bool withLocations = true);

} // namespace syntra