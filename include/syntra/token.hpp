#pragma once

#include <string>
#include "syntra/diagnostics.hpp"

namespace syntra {

/// Enumerates all token kinds recognized by the lexer.
enum class TokenKind {
    EndOfFile,      ///< End of input

    Identifier,     ///< user-defined names
    IntegerLiteral, ///< e.g. 42
    FloatLiteral,   ///< e.g. 1.23, 1e-3
    StringLiteral,  ///< e.g. "mnist.csv"

    // Keywords
    KwDataset,
    KwModel,
    KwExperiment,
    KwConfig,
    KwTrue,
    KwFalse,

    // Types (keywords)
    KwImageDataset,
    KwTabularDataset,
    KwTorchModel,
    KwMetrics,

    // Punctuation
    LBrace,   // {
    RBrace,   // }
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]
    Comma,    // ,
    Colon,    // :
    Semicolon,// ;
    Arrow,    // ->
    Dot,      // .

    // Operators / assignment / misc
    Equal,    // =
};

/// A single lexed token with its source location and raw lexeme.
struct Token {
    TokenKind kind {TokenKind::EndOfFile};
    std::string lexeme;      ///< Original text (for identifiers & literals).
    SourceLocation location; ///< Start location of the token.

    Token() = default;

    Token(TokenKind k, std::string lx, const SourceLocation& loc)
        : kind(k), lexeme(std::move(lx)), location(loc) {}
};

/// Utility to convert a TokenKind to a human-readable string.
/// Useful for debugging and error messages.
inline const char* tokenKindToString(TokenKind kind) {
    switch (kind) {
    case TokenKind::EndOfFile:      return "EndOfFile";
    case TokenKind::Identifier:     return "Identifier";
    case TokenKind::IntegerLiteral: return "IntegerLiteral";
    case TokenKind::FloatLiteral:   return "FloatLiteral";
    case TokenKind::StringLiteral:  return "StringLiteral";

    case TokenKind::KwDataset:      return "dataset";
    case TokenKind::KwModel:        return "model";
    case TokenKind::KwExperiment:   return "experiment";
    case TokenKind::KwConfig:       return "config";
    case TokenKind::KwTrue:         return "true";
    case TokenKind::KwFalse:        return "false";

    case TokenKind::KwImageDataset: return "ImageDataset";
    case TokenKind::KwTabularDataset:return "TabularDataset";
    case TokenKind::KwTorchModel:   return "TorchModel";
    case TokenKind::KwMetrics:      return "Metrics";

    case TokenKind::LBrace:   return "{";
    case TokenKind::RBrace:   return "}";
    case TokenKind::LParen:   return "(";
    case TokenKind::RParen:   return ")";
    case TokenKind::LBracket: return "[";
    case TokenKind::RBracket: return "]";
    case TokenKind::Comma:    return ",";
    case TokenKind::Colon:    return ":";
    case TokenKind::Semicolon:return ";";
    case TokenKind::Arrow:    return "->";
    case TokenKind::Dot:      return ".";

    case TokenKind::Equal:    return "=";
    }
    return "<unknown-token-kind>";
}

} // namespace syntra
