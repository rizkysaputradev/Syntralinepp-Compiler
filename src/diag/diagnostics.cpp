#include "syntra/diagnostics.hpp"

namespace syntra {

bool DiagnosticsEngine::hasErrors() const {
    for (const auto& d : diagnostics_) {
        if (d.level == DiagnosticLevel::Error) {
            return true;
        }
    }
    return false;
}

void DiagnosticsEngine::printToStdErr() const {
    for (const auto& d : diagnostics_) {
        const char* levelStr = nullptr;
        switch (d.level) {
        case DiagnosticLevel::Info:    levelStr = "info";    break;
        case DiagnosticLevel::Warning: levelStr = "warning"; break;
        case DiagnosticLevel::Error:   levelStr = "error";   break;
        }

        std::cerr << d.location.file << ":"
                  << d.location.line << ":"
                  << d.location.column << ": "
                  << levelStr << ": "
                  << d.message << "\n";
    }
}

} // namespace syntra
