#include "syntra/syntra_config.hpp"

#include <cstdlib>  // std::getenv
#include <iostream> // std::ostream
#include <string>   // std::string

namespace syntra {

namespace {

inline std::string getEnvOr(const char *name, const std::string &fallback) {
    if (!name) {
        return fallback;
    }
    if (const char *val = std::getenv(name)) {
        if (*val != '\0') {
            return std::string(val);
        }
    }
    return fallback;
}

inline bool parseBoolEnv(const char *name, bool fallback) {
    if (!name) {
        return fallback;
    }
    const char *val = std::getenv(name);
    if (!val) {
        return fallback;
    }

    std::string s(val);
    // Normalize to lowercase
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }

    if (s == "1" || s == "true" || s == "yes" || s == "on") {
        return true;
    }
    if (s == "0" || s == "false" || s == "no" || s == "off") {
        return false;
    }
    return fallback;
}

std::string normalizeBackendName(const std::string &raw) {
    std::string s = raw;
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    if (s == "simulate" || s == "simulation") {
        return "simulate";
    }
    if (s == "pytorch" || s == "torch") {
        return "pytorch";
    }
    if (s == "jax") {
        return "jax";
    }
    // Unknown backend → default to "simulate"
    return "simulate";
}

} // namespace

SyntraConfig loadConfigFromEnv() {
    SyntraConfig cfg;

    // Backend
    {
        std::string raw =
            getEnvOr("SYNTRA_BACKEND_DEFAULT", cfg.defaultBackend);
        cfg.defaultBackend = normalizeBackendName(raw);
    }

    // Default experiment
    cfg.defaultExperiment =
        getEnvOr("SYNTRA_DEFAULT_EXPERIMENT", cfg.defaultExperiment);

    // Real datasets preference
    cfg.preferRealDatasets =
        parseBoolEnv("SYNTRA_PREFER_REAL_DATASETS", cfg.preferRealDatasets);

    // Python
    cfg.pythonInterpreter = getEnvOr("SYNTRA_PYTHON", cfg.pythonInterpreter);

    // Virtual env path
    cfg.venvPath = getEnvOr("SYNTRA_VENV_PATH", cfg.venvPath);

    // Verbose diagnostics
    cfg.verboseDiagnostics =
        parseBoolEnv("SYNTRA_VERBOSE_DIAGNOSTICS", cfg.verboseDiagnostics);

    return cfg;
}

void dumpConfig(const SyntraConfig &cfg, std::ostream &os) {
    os << "syntra:\n"
       << "  backend_default      : " << cfg.defaultBackend << "\n"
       << "  default_experiment   : ";
    if (cfg.defaultExperiment.empty()) {
        os << "\"\"";
    } else {
        os << cfg.defaultExperiment;
    }
    os << "\n"
       << "  prefer_real_datasets : "
       << (cfg.preferRealDatasets ? "true" : "false") << "\n"
       << "  python_interpreter   : " << cfg.pythonInterpreter << "\n"
       << "  venv_path            : " << cfg.venvPath << "\n"
       << "  verbose_diagnostics  : "
       << (cfg.verboseDiagnostics ? "true" : "false") << "\n";
}

} // namespace syntra
