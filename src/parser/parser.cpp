#include "syntra/parser.hpp"

#include <cctype>
#include <utility>

namespace syntra {

Parser::Parser(const std::vector<Token>& tokens, DiagnosticsEngine& diag)
    : tokens_(tokens), diag_(diag) {}

const Token& Parser::current() const {
    if (index_ >= tokens_.size()) {
        return tokens_.back();
    }
    return tokens_[index_];
}

const Token& Parser::peek(std::size_t offset) const {
    if (index_ + offset >= tokens_.size()) {
        return tokens_.back();
    }
    return tokens_[index_ + offset];
}

bool Parser::isAtEnd() const {
    return current().kind == TokenKind::EndOfFile;
}

bool Parser::check(TokenKind kind) const {
    if (isAtEnd()) return false;
    return current().kind == kind;
}

bool Parser::match(TokenKind kind) {
    if (check(kind)) {
        ++index_;
        return true;
    }
    return false;
}

const Token& Parser::consume(TokenKind kind, const std::string& message) {
    if (check(kind)) {
        return tokens_[index_++];
    }

    SourceLocation loc = current().location;
    diag_.error(loc, message);
    // Return current token as a best-effort; caller should consider error state.
    return current();
}

bool Parser::checkIdentifierLexeme(const std::string& name) const {
    if (isAtEnd()) return false;
    if (current().kind != TokenKind::Identifier) return false;
    return current().lexeme == name;
}

bool Parser::matchIdentifierLexeme(const std::string& name) {
    if (checkIdentifierLexeme(name)) {
        ++index_;
        return true;
    }
    return false;
}

// ============================================================================
// Program / top-level
// ============================================================================

Program Parser::parseProgram() {
    Program prog;

    while (!isAtEnd()) {
        parseTopLevelDecl(prog);
        // In case of error, try to continue parsing more decls.
    }

    return prog;
}

void Parser::parseTopLevelDecl(Program& prog) {
    const Token& tok = current();
    switch (tok.kind) {
    case TokenKind::KwDataset: {
        Token datasetTok = tok;
        ++index_;
        DatasetDecl d = parseDatasetDecl(datasetTok);
        prog.datasets.push_back(std::move(d));
        break;
    }
    case TokenKind::KwModel: {
        Token modelTok = tok;
        ++index_;
        ModelDecl m = parseModelDecl(modelTok);
        prog.models.push_back(std::move(m));
        break;
    }
    case TokenKind::KwExperiment: {
        Token expTok = tok;
        ++index_;
        ExperimentDecl e = parseExperimentDecl(expTok);
        prog.experiments.push_back(std::move(e));
        break;
    }
    case TokenKind::EndOfFile:
        // Nothing to do
        break;
    default: {
        // Unexpected top-level token
        diag_.error(tok.location, "expected 'dataset', 'model', or 'experiment' at top level");
        synchronize();
        break;
    }
    }
}

// ============================================================================
// DatasetDecl
// ============================================================================

DatasetDecl Parser::parseDatasetDecl(const Token& datasetToken) {
    DatasetDecl decl;
    decl.loc = datasetToken.location;

    const Token& nameTok = consume(TokenKind::Identifier, "expected dataset name after 'dataset'");
    decl.name = nameTok.lexeme;

    consume(TokenKind::Colon, "expected ':' after dataset name");

    const Token& typeTok = current();
    if (typeTok.kind == TokenKind::KwImageDataset || typeTok.kind == TokenKind::KwTabularDataset
        || typeTok.kind == TokenKind::Identifier) {
        decl.datasetType = typeTok.lexeme;
        ++index_;
    } else {
        diag_.error(typeTok.location, "expected dataset type (e.g. ImageDataset)");
    }

    consume(TokenKind::LBrace, "expected '{' to start dataset body");

    decl.fields = parseFieldBlock();

    return decl;
}

// ============================================================================
// ModelDecl
// ============================================================================

ModelDecl Parser::parseModelDecl(const Token& modelToken) {
    ModelDecl decl;
    decl.loc = modelToken.location;

    const Token& nameTok = consume(TokenKind::Identifier, "expected model name after 'model'");
    decl.name = nameTok.lexeme;

    consume(TokenKind::Colon, "expected ':' after model name");

    const Token& typeTok = current();
    if (typeTok.kind == TokenKind::KwTorchModel || typeTok.kind == TokenKind::Identifier) {
        decl.modelType = typeTok.lexeme;
        ++index_;
    } else {
        diag_.error(typeTok.location, "expected model type (e.g. TorchModel)");
    }

    consume(TokenKind::LBrace, "expected '{' to start model body");

    decl.fields = parseFieldBlock();

    return decl;
}

// ============================================================================
// Field block: { name = expr ... }
// ============================================================================

std::vector<NamedField> Parser::parseFieldBlock() {
    std::vector<NamedField> fields;

    while (!isAtEnd() && !check(TokenKind::RBrace)) {
        const Token& nameTok = consume(TokenKind::Identifier, "expected field name in block");
        std::string fieldName = nameTok.lexeme;

        consume(TokenKind::Equal, "expected '=' after field name");

        ExprPtr value = parseExpression();

        NamedField field;
        field.name = fieldName;
        field.value = std::move(value);
        field.loc = nameTok.location;
        fields.push_back(std::move(field));

        // No semicolons required; newline separation is implicit in the source.
        // We simply loop until we see '}'.
    }

    consume(TokenKind::RBrace, "expected '}' at end of block");

    return fields;
}

// ============================================================================
// ExperimentDecl
// ============================================================================

ExperimentDecl Parser::parseExperimentDecl(const Token& experimentToken) {
    ExperimentDecl decl;
    decl.loc = experimentToken.location;

    const Token& nameTok = consume(TokenKind::Identifier, "expected experiment name after 'experiment'");
    decl.name = nameTok.lexeme;

    consume(TokenKind::LParen, "expected '(' after experiment name");

    const Token& paramTok = consume(TokenKind::Identifier, "expected dataset parameter name");
    decl.datasetParam = paramTok.lexeme;

    consume(TokenKind::RParen, "expected ')' after experiment dataset parameter");

    consume(TokenKind::Arrow, "expected '->' after experiment parameter list");

    const Token& resultTypeTok = current();
    if (resultTypeTok.kind == TokenKind::KwMetrics || resultTypeTok.kind == TokenKind::Identifier) {
        decl.resultType = resultTypeTok.lexeme;
        ++index_;
    } else {
        diag_.error(resultTypeTok.location, "expected result type after '->' (e.g. Metrics)");
    }

    consume(TokenKind::LBrace, "expected '{' to start experiment body");

    decl.body = parseExperimentBody();

    return decl;
}

// experiment body: sequence of let/return statements until '}'
std::vector<StmtPtr> Parser::parseExperimentBody() {
    std::vector<StmtPtr> body;

    while (!isAtEnd() && !check(TokenKind::RBrace)) {
        StmtPtr stmt = parseStatement();
        if (stmt) {
            body.push_back(std::move(stmt));
        }
    }

    consume(TokenKind::RBrace, "expected '}' at end of experiment body");

    return body;
}

StmtPtr Parser::parseStatement() {
    const Token& tok = current();

    // Pseudo-keywords: "let" and "return" as identifiers
    if (checkIdentifierLexeme("let")) {
        Token letTok = tok;
        ++index_;
        return parseLetStatement(letTok);
    }

    if (checkIdentifierLexeme("return")) {
        Token retTok = tok;
        ++index_;
        return parseReturnStatement(retTok);
    }

    diag_.error(tok.location, "expected 'let' or 'return' in experiment body");
    synchronize();
    return nullptr;
}

// let <name> = <expr>
std::unique_ptr<LetStmt> Parser::parseLetStatement(const Token& letToken) {
    const Token& nameTok = consume(TokenKind::Identifier, "expected variable name after 'let'");
    std::string varName = nameTok.lexeme;

    consume(TokenKind::Equal, "expected '=' after variable name in let-statement");

    ExprPtr expr = parseExpression();

    return std::make_unique<LetStmt>(varName, std::move(expr), letToken.location);
}

// return { field = expr, ... }
std::unique_ptr<ReturnStmt> Parser::parseReturnStatement(const Token& returnToken) {
    consume(TokenKind::LBrace, "expected '{' after 'return'");

    std::vector<NamedField> fields = parseNamedFieldListInBraces();
    return std::make_unique<ReturnStmt>(std::move(fields), returnToken.location);
}

std::vector<NamedField> Parser::parseNamedFieldListInBraces() {
    std::vector<NamedField> fields;

    while (!isAtEnd() && !check(TokenKind::RBrace)) {
        const Token& nameTok = consume(TokenKind::Identifier, "expected field name in return block");
        std::string fieldName = nameTok.lexeme;

        consume(TokenKind::Equal, "expected '=' after field name in return block");

        ExprPtr value = parseExpression();

        NamedField field;
        field.name = fieldName;
        field.value = std::move(value);
        field.loc = nameTok.location;
        fields.push_back(std::move(field));

        // Optional comma
        if (check(TokenKind::Comma)) {
            ++index_;
        }
    }

    consume(TokenKind::RBrace, "expected '}' at end of return block");

    return fields;
}

// ============================================================================
// Expressions
// ============================================================================

// We implement a simple expression grammar sufficient for v0.1:
//
//   expr := primary postfix* [ "with" config_block ]?
//
// postfix:
//   '.' Identifier
//   '(' args? ')'
//
// config_block:
//   '{' name '=' expr (',' name '=' expr)* '}'
//
// primary:
//   Identifier
//   IntegerLiteral
//   FloatLiteral
//   StringLiteral
//   true / false
//   '(' expr (',' expr)* ')'    // tuple or parenthesized
//   '[' expr (',' expr)* ']'    // list
//
// More complex precedence can be added later if needed.

ExprPtr Parser::parseExpression() {
    ExprPtr base = parsePrimary();
    base = parsePostfix(std::move(base));

    // Optional "with" config
    if (checkIdentifierLexeme("with")) {
        Token withTok = current();
        ++index_;

        consume(TokenKind::LBrace, "expected '{' after 'with'");

        std::vector<NamedField> fields = parseNamedFieldListInBraces();
        // Note: parseNamedFieldListInBraces already consumes the closing '}'.

        auto withExpr = std::make_unique<WithConfigExpr>(std::move(base), std::move(fields), withTok.location);
        base = std::move(withExpr);
    }

    return base;
}

ExprPtr Parser::parsePrimary() {
    const Token& tok = current();

    switch (tok.kind) {
    case TokenKind::Identifier: {
        SourceLocation loc = tok.location;
        std::string name = tok.lexeme;
        ++index_;
        return std::make_unique<IdentifierExpr>(name, loc);
    }
    case TokenKind::IntegerLiteral: {
        SourceLocation loc = tok.location;
        std::string val = tok.lexeme;
        ++index_;
        return std::make_unique<IntegerLiteralExpr>(val, loc);
    }
    case TokenKind::FloatLiteral: {
        SourceLocation loc = tok.location;
        std::string val = tok.lexeme;
        ++index_;
        return std::make_unique<FloatLiteralExpr>(val, loc);
    }
    case TokenKind::StringLiteral: {
        SourceLocation loc = tok.location;
        std::string val = tok.lexeme;
        ++index_;
        return std::make_unique<StringLiteralExpr>(val, loc);
    }
    case TokenKind::KwTrue: {
        SourceLocation loc = tok.location;
        ++index_;
        return std::make_unique<BoolLiteralExpr>(true, loc);
    }
    case TokenKind::KwFalse: {
        SourceLocation loc = tok.location;
        ++index_;
        return std::make_unique<BoolLiteralExpr>(false, loc);
    }
    case TokenKind::LParen: {
        Token lparen = tok;
        ++index_;
        return parseTupleOrParenExpr(lparen);
    }
    case TokenKind::LBracket: {
        Token lbracket = tok;
        ++index_;
        return parseListExpr(lbracket);
    }
    default:
        diag_.error(tok.location, "unexpected token in expression");
        // Return a dummy identifier to keep AST shape usable.
        ++index_;
        return std::make_unique<IdentifierExpr>("<error>", tok.location);
    }
}

ExprPtr Parser::parsePostfix(ExprPtr base) {
    while (!isAtEnd()) {
        if (match(TokenKind::Dot)) {
            const Token& nameTok = consume(TokenKind::Identifier, "expected identifier after '.'");
            SourceLocation loc = nameTok.location;
            std::string memberName = nameTok.lexeme;
            base = std::make_unique<MemberAccessExpr>(std::move(base), memberName, loc);
        } else if (match(TokenKind::LParen)) {
            // Argument list
            std::vector<ExprPtr> args;

            if (!check(TokenKind::RParen)) {
                do {
                    ExprPtr arg = parseExpression();
                    args.push_back(std::move(arg));
                } while (match(TokenKind::Comma));
            }

            const Token& rparen = consume(TokenKind::RParen, "expected ')' after argument list");
            SourceLocation loc = rparen.location;
            base = std::make_unique<CallExpr>(std::move(base), std::move(args), loc);
        } else {
            break;
        }
    }
    return base;
}

ExprPtr Parser::parseTupleOrParenExpr(const Token& lparen) {
    std::vector<ExprPtr> elems;

    if (!check(TokenKind::RParen)) {
        do {
            ExprPtr e = parseExpression();
            elems.push_back(std::move(e));
        } while (match(TokenKind::Comma));
    }

    const Token& rparen = consume(TokenKind::RParen, "expected ')' to close tuple/paren expression");

    if (elems.size() == 1) {
        // Just a parenthesized expression
        return std::move(elems[0]);
    }

    return std::make_unique<TupleExpr>(std::move(elems), lparen.location);
}

ExprPtr Parser::parseListExpr(const Token& lbracket) {
    std::vector<ExprPtr> elems;

    if (!check(TokenKind::RBracket)) {
        do {
            ExprPtr e = parseExpression();
            elems.push_back(std::move(e));
        } while (match(TokenKind::Comma));
    }

    consume(TokenKind::RBracket, "expected ']' to close list literal");
    return std::make_unique<ListExpr>(std::move(elems), lbracket.location);
}

// ============================================================================
// Error recovery
// ============================================================================

void Parser::synchronize() {
    // A very simple error recovery strategy:
    // Skip tokens until we reach a likely good restart point:
    //  - a top-level keyword (dataset/model/experiment)
    //  - a closing brace
    while (!isAtEnd()) {
        if (check(TokenKind::KwDataset) ||
            check(TokenKind::KwModel)   ||
            check(TokenKind::KwExperiment) ||
            check(TokenKind::RBrace)) {
            return;
        }
        ++index_;
    }
}

} // namespace syntra
