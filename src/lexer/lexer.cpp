#include "syntra/lexer.hpp"

#include <cctype>

namespace syntra {

Lexer::Lexer(std::string fileName,
             std::string source,
             DiagnosticsEngine& diag)
    : file_(std::move(fileName)),
      src_(std::move(source)),
      diag_(diag) {}

char Lexer::peek(std::size_t offset) const {
    if (index_ + offset >= src_.size()) {
        return '\0';
    }
    return src_[index_ + offset];
}

char Lexer::get() {
    if (isAtEnd()) {
        return '\0';
    }
    char c = src_[index_++];
    if (c == '\n') {
        ++line_;
        column_ = 1;
    } else {
        ++column_;
    }
    return c;
}

bool Lexer::isAtEnd() const {
    return index_ >= src_.size();
}

void Lexer::skipWhitespaceAndComments() {
    while (!isAtEnd()) {
        char c = peek();
        // Whitespace
        if (std::isspace(static_cast<unsigned char>(c))) {
            get();
            continue;
        }
        // Line comments: //
        if (c == '/' && peek(1) == '/') {
            get(); // '/'
            get(); // '/'
            while (!isAtEnd() && peek() != '\n') {
                get();
            }
            continue;
        }
        // Block comments (/* ... */) — optional, but nice to have
        if (c == '/' && peek(1) == '*') {
            get(); // '/'
            get(); // '*'
            while (!isAtEnd()) {
                if (peek() == '*' && peek(1) == '/') {
                    get(); // '*'
                    get(); // '/'
                    break;
                }
                get();
            }
            continue;
        }
        break; // Not whitespace or comment
    }
}

Token Lexer::makeToken(TokenKind kind, const std::string& lexeme,
                       std::size_t startLine, std::size_t startCol) {
    SourceLocation loc{file_, startLine, startCol};
    return Token{kind, lexeme, loc};
}

Token Lexer::errorToken(const std::string& msg) {
    SourceLocation loc{file_, line_, column_};
    diag_.error(loc, msg);
    return Token{TokenKind::EndOfFile, "", loc};
}

Token Lexer::lexIdentifierOrKeyword() {
    std::size_t startIndex = index_ - 1;
    std::size_t startLine = line_;
    std::size_t startCol = column_ - 1;

    while (!isAtEnd()) {
        char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            get();
        } else {
            break;
        }
    }

    std::string text = src_.substr(startIndex, index_ - startIndex);

    // Recognize keywords / type keywords
    if (text == "dataset")       return makeToken(TokenKind::KwDataset, text, startLine, startCol);
    if (text == "model")         return makeToken(TokenKind::KwModel, text, startLine, startCol);
    if (text == "experiment")    return makeToken(TokenKind::KwExperiment, text, startLine, startCol);
    if (text == "config")        return makeToken(TokenKind::KwConfig, text, startLine, startCol);
    if (text == "true")          return makeToken(TokenKind::KwTrue, text, startLine, startCol);
    if (text == "false")         return makeToken(TokenKind::KwFalse, text, startLine, startCol);

    if (text == "ImageDataset")  return makeToken(TokenKind::KwImageDataset, text, startLine, startCol);
    if (text == "TabularDataset")return makeToken(TokenKind::KwTabularDataset, text, startLine, startCol);
    if (text == "TorchModel")    return makeToken(TokenKind::KwTorchModel, text, startLine, startCol);
    if (text == "Metrics")       return makeToken(TokenKind::KwMetrics, text, startLine, startCol);

    // Otherwise, user-defined identifier
    return makeToken(TokenKind::Identifier, text, startLine, startCol);
}

Token Lexer::lexNumberLiteral(char firstChar) {
    std::size_t startIndex = index_ - 1;
    std::size_t startLine = line_;
    std::size_t startCol = column_ - 1;

    bool isFloat = false;

    // integer part
    while (!isAtEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
        get();
    }

    // decimal part
    if (peek() == '.' && std::isdigit(static_cast<unsigned char>(peek(1)))) {
        isFloat = true;
        get(); // consume '.'
        while (!isAtEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
            get();
        }
    }

    // optional exponent part (e.g. 1e-3)
    if (peek() == 'e' || peek() == 'E') {
        isFloat = true;
        get(); // 'e' or 'E'

        if (peek() == '+' || peek() == '-') {
            get();
        }

        if (!std::isdigit(static_cast<unsigned char>(peek()))) {
            // malformed exponent, but we still accept token and report warning
            SourceLocation loc{file_, line_, column_};
            diag_.warn(loc, "malformed exponent in float literal");
        } else {
            while (!isAtEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
                get();
            }
        }
    }

    std::string text = src_.substr(startIndex, index_ - startIndex);
    TokenKind kind = isFloat ? TokenKind::FloatLiteral : TokenKind::IntegerLiteral;
    return makeToken(kind, text, startLine, startCol);
}

Token Lexer::lexStringLiteral() {
    std::size_t startLine = line_;
    std::size_t startCol = column_; // we already consumed the opening quote

    std::string buffer;
    while (!isAtEnd()) {
        char c = get();
        if (c == '"') {
            // closing quote
            SourceLocation loc{file_, startLine, startCol - 1};
            return Token{TokenKind::StringLiteral, buffer, loc};
        }
        if (c == '\\') {
            // escape sequence
            if (isAtEnd()) break;
            char esc = get();
            switch (esc) {
            case 'n': buffer.push_back('\n'); break;
            case 't': buffer.push_back('\t'); break;
            case 'r': buffer.push_back('\r'); break;
            case '\\': buffer.push_back('\\'); break;
            case '"': buffer.push_back('"'); break;
            default:
                diag_.warn(SourceLocation{file_, line_, column_},
                           std::string("unknown escape sequence: \\") + esc);
                buffer.push_back(esc);
                break;
            }
        } else {
            buffer.push_back(c);
        }
    }

    // If we reach here, string was unterminated
    return errorToken("unterminated string literal");
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    while (true) {
        skipWhitespaceAndComments();

        if (isAtEnd()) {
            SourceLocation loc{file_, line_, column_};
            tokens.emplace_back(TokenKind::EndOfFile, "", loc);
            break;
        }

        char c = get();
        std::size_t startLine = line_;
        std::size_t startCol = column_ - 1;

        // Identifiers / keywords
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            // We already consumed the first character of the identifier in `c`,
            // and `index_` now points to the next character. `lexIdentifierOrKeyword`
            // assumes exactly this state and will treat `index_ - 1` as the start.
            Token t = lexIdentifierOrKeyword();
            tokens.push_back(std::move(t));
            if (diag_.hasErrors()) break;
            continue;
        }

        // Numbers
        if (std::isdigit(static_cast<unsigned char>(c))) {
            Token t = lexNumberLiteral(c);
            tokens.push_back(std::move(t));
            if (diag_.hasErrors()) break;
            continue;
        }

        // Strings
        if (c == '"') {
            Token t = lexStringLiteral();
            tokens.push_back(std::move(t));
            if (diag_.hasErrors()) break;
            continue;
        }

        // Punctuation / operators
        switch (c) {
        case '{': tokens.push_back(makeToken(TokenKind::LBrace, "{", startLine, startCol)); break;
        case '}': tokens.push_back(makeToken(TokenKind::RBrace, "}", startLine, startCol)); break;
        case '(': tokens.push_back(makeToken(TokenKind::LParen, "(", startLine, startCol)); break;
        case ')': tokens.push_back(makeToken(TokenKind::RParen, ")", startLine, startCol)); break;
        case '[': tokens.push_back(makeToken(TokenKind::LBracket, "[", startLine, startCol)); break;
        case ']': tokens.push_back(makeToken(TokenKind::RBracket, "]", startLine, startCol)); break;
        case ',': tokens.push_back(makeToken(TokenKind::Comma, ",", startLine, startCol)); break;
        case ':': tokens.push_back(makeToken(TokenKind::Colon, ":", startLine, startCol)); break;
        case ';': tokens.push_back(makeToken(TokenKind::Semicolon, ";", startLine, startCol)); break;
        case '.': tokens.push_back(makeToken(TokenKind::Dot, ".", startLine, startCol)); break;
        case '=': tokens.push_back(makeToken(TokenKind::Equal, "=", startLine, startCol)); break;
        case '-':
            if (peek() == '>') {
                get(); // consume '>'
                tokens.push_back(makeToken(TokenKind::Arrow, "->", startLine, startCol));
            } else {
                diag_.error(SourceLocation{file_, startLine, startCol},
                            "unexpected character '-' (only '->' is allowed)");
                tokens.emplace_back(TokenKind::EndOfFile, "", SourceLocation{file_, startLine, startCol});
            }
            break;
        default: {
            // Skip unknown characters but report an error
            std::string msg = "unexpected character '";
            msg.push_back(c);
            msg.push_back('\'');
            diag_.error(SourceLocation{file_, startLine, startCol}, msg);
            tokens.emplace_back(TokenKind::EndOfFile, "", SourceLocation{file_, startLine, startCol});
            break;
        }
        }

        if (diag_.hasErrors()) {
            break;
        }
    }

    return tokens;
}

} // namespace syntra
