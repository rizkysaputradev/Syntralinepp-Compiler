#pragma once

#include <string>
#include <vector>

#include "syntra/token.hpp"
#include "syntra/diagnostics.hpp"

namespace syntra {

/// Lexical analyzer for SyntraLine++ source code.
///
/// The Lexer takes a source buffer and produces a flat vector of Tokens.
/// It does not perform any semantic validation beyond:
///   - recognizing keywords,
///   - validating numeric literal shapes.
/// Any problems are reported via DiagnosticsEngine.
class Lexer {
public:
    Lexer(std::string fileName,
          std::string source,
          DiagnosticsEngine& diag);

    /// Tokenize the entire input and return a vector of tokens.
    /// The last token is always EndOfFile unless a fatal lexing error occurs.
    std::vector<Token> tokenize();

private:
    // Core helpers
    char peek(std::size_t offset = 0) const;
    char get();
    bool isAtEnd() const;

    void skipWhitespaceAndComments();

    Token lexIdentifierOrKeyword();
    Token lexNumberLiteral(char firstChar);
    Token lexStringLiteral();

    Token makeToken(TokenKind kind, const std::string& lexeme,
                    std::size_t startLine, std::size_t startCol);

    Token errorToken(const std::string& msg);

    // State
    std::string file_;
    std::string src_;
    std::size_t index_ {0};
    std::size_t line_ {1};
    std::size_t column_ {1};

    DiagnosticsEngine& diag_;
};

} // namespace syntra
