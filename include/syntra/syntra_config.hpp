#pragma once

// ============================================================================
// SyntraLine++ Compiler Configuration
// ----------------------------------------------------------------------------
// This header collects small compile-time configuration switches and
// project-wide constants. It also defines a small runtime configuration
// struct that can be loaded from environment variables.
// ============================================================================

#include <iosfwd> // std::ostream forward decl
#include <string>

namespace syntra {

/// Current semantic version of the compiler as seen in --version.
inline constexpr const char *SYNTRA_VERSION = "0.1.0";

/// Default file extension for SyntraLine++ source files.
inline constexpr const char *SYNTRA_DEFAULT_EXTENSION = ".syntra";

/// Name used in diagnostics when no real file name is available.
inline constexpr const char *SYNTRA_STDIN_FILENAME = "<stdin>";

/// ---------------------------------------------------------------------------
/// Runtime configuration
/// ---------------------------------------------------------------------------
/// Global configuration for SyntraLine++ execution.
///
/// Values are typically populated from environment variables, with sensible
/// defaults so that `syntra` works even without any user config.
struct SyntraConfig {
    /// Default backend to use when the user does not explicitly request one.
    /// Typical values:
    ///   - "simulate"
    ///   - "pytorch"
    ///   - "jax"
    ///
    /// For now, we do not automatically route on this in main(); it is mainly
    /// surfaced via --dump-config and available for future logic.
    std::string defaultBackend{"simulate"};

    /// Default experiment name to run when none is specified.
    /// If empty, the compiler/runtime should fall back to the first
    /// experiment defined in the module.
    std::string defaultExperiment;

    /// If true, the runtime/backends are *allowed* to prefer real datasets
    /// (e.g., TorchVision / TFDS) over synthetic stubs, when such backends
    /// are available. If false, they may choose to always use synthetic
    /// random data.
    bool preferRealDatasets{false};

    /// Name/path of the Python interpreter to use when executing generated
    /// PyTorch/JAX scripts (e.g., in docs, scripts, or tooling).
    /// This does not affect the compiler itself, but is useful for tooling.
    std::string pythonInterpreter{"python3"};

    /// Path to the virtual environment where SyntraLine++ Python
    /// dependencies (torch, jax, pytest, etc.) are installed.
    /// For example: ".syntra-venv".
    std::string venvPath{".syntra-venv"};

    /// If true, the compiler and tooling may choose to emit more verbose
    /// diagnostics (e.g., extra notes, debug info). This does not change
    /// correctness, only verbosity.
    bool verboseDiagnostics{false};
};

/// Load a SyntraConfig from the current process environment.
///
/// Recognized variables (all optional):
///   - SYNTRA_BACKEND_DEFAULT
///       One of: "simulate", "pytorch", "jax".
///       Default: "simulate" if unset or unrecognized.
///
///   - SYNTRA_DEFAULT_EXPERIMENT
///       Name of the default experiment to run when not specified.
///       Default: empty string (meaning "first experiment").
///
///   - SYNTRA_PREFER_REAL_DATASETS
///       "1", "true", "yes", "on"  => preferRealDatasets = true
///       "0", "false", "no", "off" => preferRealDatasets = false
///       Unset                     => false
///
///   - SYNTRA_PYTHON
///       Path/name of Python interpreter. Default "python3".
///
///   - SYNTRA_VENV_PATH
///       Path to the Python virtual environment. Default ".syntra-venv".
///
///   - SYNTRA_VERBOSE_DIAGNOSTICS
///       Same bool conventions as SYNTRA_PREFER_REAL_DATASETS.
///       Default: false.
SyntraConfig loadConfigFromEnv();

/// Dump the given configuration in a human-readable form.
///
/// Example output:
///
/// syntra:
///   backend_default      : simulate
///   default_experiment   : ""
///   prefer_real_datasets : false
///   python_interpreter   : python3
///   venv_path            : .syntra-venv
///   verbose_diagnostics  : false
///
/// Used by the `--dump-config` CLI option.
void dumpConfig(const SyntraConfig &cfg, std::ostream &os);

} // namespace syntra