#pragma once

#include <string>
#include <vector>
#include <iostream>

namespace syntra {

struct SourceLocation {
    std::string file;
    std::size_t line {1};
    std::size_t column {1};
};

enum class DiagnosticLevel {
    Info,
    Warning,
    Error,
};

struct Diagnostic {
    DiagnosticLevel level;
    SourceLocation location;
    std::string message;
};

class DiagnosticsEngine {
public:
    void report(DiagnosticLevel level,
                const SourceLocation& loc,
                const std::string& msg) {
        diagnostics_.push_back(Diagnostic{level, loc, msg});
    }

    void info(const SourceLocation& loc, const std::string& msg) {
        report(DiagnosticLevel::Info, loc, msg);
    }

    void warn(const SourceLocation& loc, const std::string& msg) {
        report(DiagnosticLevel::Warning, loc, msg);
    }

    void error(const SourceLocation& loc, const std::string& msg) {
        report(DiagnosticLevel::Error, loc, msg);
    }

    /// Returns true if any diagnostics of level Error were recorded.
    bool hasErrors() const;

    /// Print all diagnostics to stderr in a human-readable format.
    void printToStdErr() const;

    const std::vector<Diagnostic>& diagnostics() const {
        return diagnostics_;
    }

private:
    std::vector<Diagnostic> diagnostics_;
};

} // namespace syntra
