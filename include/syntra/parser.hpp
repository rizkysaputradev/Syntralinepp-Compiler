#pragma once

#include <vector>

#include "syntra/ast.hpp"
#include "syntra/token.hpp"
#include "syntra/diagnostics.hpp"

namespace syntra {

/// Recursive-descent parser for SyntraLine++ v0.1.
///
/// Takes a flat vector of tokens (produced by Lexer) and builds a Program AST.
/// On errors, it reports diagnostics and returns a partially constructed Program.
class Parser {
public:
    Parser(const std::vector<Token>& tokens, DiagnosticsEngine& diag);

    /// Parse the full input into a Program.
    Program parseProgram();

private:
    // Current token helpers
    const Token& current() const;
    const Token& peek(std::size_t offset) const;
    bool isAtEnd() const;

    bool check(TokenKind kind) const;
    bool match(TokenKind kind);
    const Token& consume(TokenKind kind, const std::string& message);

    // Pseudo-keyword handling by lexeme, e.g. "let", "return", "with"
    bool checkIdentifierLexeme(const std::string& name) const;
    bool matchIdentifierLexeme(const std::string& name);

    // Top-level parsing
    void parseTopLevelDecl(Program& prog);
    DatasetDecl parseDatasetDecl(const Token& datasetToken);
    ModelDecl parseModelDecl(const Token& modelToken);
    ExperimentDecl parseExperimentDecl(const Token& experimentToken);

    // Common helpers: parse a block of "field = expr" pairs until '}'.
    std::vector<NamedField> parseFieldBlock();

    // Experiment body parsing
    std::vector<StmtPtr> parseExperimentBody();
    StmtPtr parseStatement();
    std::unique_ptr<LetStmt> parseLetStatement(const Token& letToken);
    std::unique_ptr<ReturnStmt> parseReturnStatement(const Token& returnToken);
    std::vector<NamedField> parseNamedFieldListInBraces();

    // Expressions
    ExprPtr parseExpression();
    ExprPtr parsePrimary();
    ExprPtr parsePostfix(ExprPtr base);

    ExprPtr parseTupleOrParenExpr(const Token& lparen);
    ExprPtr parseListExpr(const Token& lbracket);
    WithConfigExpr* wrapWithConfigIfPresent(ExprPtr& base); // optional helper

    // Error handling
    void synchronize();

    const std::vector<Token>& tokens_;
    DiagnosticsEngine& diag_;
    std::size_t index_ {0};
};

} // namespace syntra
