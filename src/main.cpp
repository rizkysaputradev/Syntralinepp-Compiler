#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "syntra/ast.hpp"
#include "syntra/backend_jax.hpp"
#include "syntra/backend_pytorch.hpp"
#include "syntra/backend_runtime.hpp"
#include "syntra/diagnostics.hpp"
#include "syntra/ir.hpp"
#include "syntra/ir_execution.hpp"
#include "syntra/ir_pipeline.hpp"
#include "syntra/lexer.hpp"
#include "syntra/optimizer.hpp"
#include "syntra/parser.hpp"
#include "syntra/pass_manager.hpp"
#include "syntra/sema.hpp"
#include "syntra/syntra_config.hpp"
#include "syntra/token.hpp"

using namespace syntra;

static std::string readFileToString(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void printUsage() {
    std::cerr
        << "SyntraLine++ compiler (syntra) v" << SYNTRA_VERSION << "\n\n"
        << "Usage:\n"
        << "  syntra [options] <input.syntra>\n\n"
        << "Options:\n"
        << "  --dump-tokens           Lex the file and print tokens (debug).\n"
        << "  --dump-ast              Parse and dump the AST (after sema).\n"
        << "  --dump-ir               Parse + analyze and dump pipeline IR.\n"
        << "  --emit-pytorch          Emit a PyTorch training script to "
           "stdout.\n"
        << "  --emit-jax              Emit a JAX-based training script to "
           "stdout.\n"
        << "  --simulate              Run the first experiment via in-process "
           "runtime.\n"
        << "  --experiment <name>     Select which experiment to run/emit "
           "(used with --simulate or --emit-pytorch).\n"
        << "  --dump-config           Show effective Syntra configuration and "
           "exit.\n"
        << "  --help                  Show this help.\n"
        << "  --version               Show version.\n";
}

static void dumpTokens(const std::vector<Token> &tokens) {
    for (const auto &tok : tokens) {
        std::cout << tok.location.line << ":" << tok.location.column << "  "
                  << tokenKindToString(tok.kind);

        if (!tok.lexeme.empty() && tok.kind != TokenKind::EndOfFile &&
            tok.kind != TokenKind::LBrace && tok.kind != TokenKind::RBrace &&
            tok.kind != TokenKind::LParen && tok.kind != TokenKind::RParen &&
            tok.kind != TokenKind::LBracket &&
            tok.kind != TokenKind::RBracket && tok.kind != TokenKind::Comma &&
            tok.kind != TokenKind::Colon && tok.kind != TokenKind::Semicolon &&
            tok.kind != TokenKind::Arrow && tok.kind != TokenKind::Dot &&
            tok.kind != TokenKind::Equal) {
            std::cout << "  [" << tok.lexeme << "]";
        }

        std::cout << "\n";
    }
}

static void dumpIR(const PipelineModule &module) {
    std::cout << "=== Pipeline IR Dump ===\n";

    std::cout << "\nDatasets (" << module.datasets.size() << "):\n";
    for (const auto &d : module.datasets) {
        std::cout << "  - " << d.name << " : " << d.typeName << "\n";
    }

    std::cout << "\nModels (" << module.models.size() << "):\n";
    for (const auto &m : module.models) {
        std::cout << "  - " << m.name << " : " << m.typeName << "\n";
    }

    std::cout << "\nExperiments (" << module.experiments.size() << "):\n";
    for (const auto &e : module.experiments) {
        std::cout << "  - " << e.name << "(" << e.datasetParam << ") -> "
                  << e.resultTypeName << "\n";

        for (const auto &op : e.ops) {
            switch (op.kind) {
            case PipelineOpKind::Train:
                std::cout << "      train " << op.train.varName << " = train("
                          << op.train.modelName << ", " << op.train.datasetName
                          << "." << op.train.splitName << ")\n";
                break;
            case PipelineOpKind::Evaluate:
                std::cout << "      eval " << op.eval.varName << " = evaluate("
                          << op.eval.modelName << ", " << op.eval.datasetName
                          << "." << op.eval.splitName << ") metrics=[";
                for (std::size_t i = 0; i < op.eval.metrics.size(); ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << op.eval.metrics[i];
                }
                std::cout << "]\n";
                break;
            }
        }

        if (!e.results.empty()) {
            std::cout << "      return {\n";
            for (const auto &rf : e.results) {
                std::cout << "          " << rf.fieldName << " = "
                          << rf.sourceVar << "." << rf.metricName << "\n";
            }
            std::cout << "      }\n";
        }
    }

    std::cout << "=== End IR Dump ===\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    bool dumpTokensFlag = false;
    bool dumpIRFlag = false;
    bool dumpASTFlag = false;
    bool emitPytorchFlag = false;
    bool emitJaxFlag = false;
    bool simulateFlag = false;
    bool dumpConfigFlag = false;

    std::string inputPath;
    std::string experimentName;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else if (arg == "--version") {
            std::cout << "SyntraLine++ compiler (syntra) v" << SYNTRA_VERSION
                      << "\n";
            return 0;
        } else if (arg == "--dump-config") {
            dumpConfigFlag = true;
        } else if (arg == "--dump-tokens") {
            dumpTokensFlag = true;
        } else if (arg == "--dump-ast") {
            dumpASTFlag = true;
        } else if (arg == "--dump-ir") {
            dumpIRFlag = true;
        } else if (arg == "--emit-pytorch") {
            emitPytorchFlag = true;
        } else if (arg == "--emit-jax") {
            emitJaxFlag = true;
        } else if (arg == "--simulate") {
            simulateFlag = true;
        } else if (arg == "--experiment") {
            if (i + 1 >= argc) {
                std::cerr << "--experiment requires a name argument\n";
                return 1;
            }
            experimentName = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << "\n";
            printUsage();
            return 1;
        } else {
            if (!inputPath.empty()) {
                std::cerr << "multiple input files are not supported yet\n";
                return 1;
            }
            inputPath = arg;
        }
    }

    // Load configuration from environment once per process.
    SyntraConfig config = loadConfigFromEnv();

    if (dumpConfigFlag) {
        dumpConfig(config, std::cout);
        return 0;
    }

    if (inputPath.empty()) {
        std::cerr << "no input file specified\n";
        printUsage();
        return 1;
    }

    if (emitPytorchFlag && emitJaxFlag) {
        std::cerr
            << "cannot use --emit-pytorch and --emit-jax at the same time\n";
        printUsage();
        return 1;
    }

    // ---------------------------------------------------------------------
    // Load Syntra config and decide which experiment name to use
    // ---------------------------------------------------------------------

    // CLI has highest priority; if CLI didn't specify --experiment
    std::string effectiveExperimentName = experimentName;

    // we fall back to config.defaultExperiment (if set).
    if (effectiveExperimentName.empty() && !config.defaultExperiment.empty()) {
        effectiveExperimentName = config.defaultExperiment;
    }

    DiagnosticsEngine diag;

    std::string source;
    try {
        source = readFileToString(inputPath);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    Lexer lexer(inputPath, source, diag);
    auto tokens = lexer.tokenize();

    if (diag.hasErrors()) {
        diag.printToStdErr();
        return 1;
    }

    if (dumpTokensFlag) {
        dumpTokens(tokens);
        return 0;
    }

    // Parse into AST
    Parser parser(tokens, diag);
    Program program = parser.parseProgram();

    if (diag.hasErrors()) {
        diag.printToStdErr();
        return 1;
    }

    // Semantic analysis
    SemanticAnalyzer sema(diag);
    sema.analyze(program);

    if (diag.hasErrors()) {
        diag.printToStdErr();
        return 1;
    }

    if (dumpASTFlag) {
        dumpAST(program, std::cout, /*withLocations=*/true);
        return 0;
    }

    // Build pipeline IR
    PipelineBuilder builder(diag);
    PipelineModule module = builder.build(program);

    if (diag.hasErrors()) {
        diag.printToStdErr();
        return 1;
    }

    // Run IR optimization / canonicalization passes
    PassManager pm;
    pm.addDefaultPasses();
    pm.run(module, diag);

    if (diag.hasErrors()) {
        diag.printToStdErr();
        return 1;
    }

    if (simulateFlag) {
        PipelineExecutor exec(diag);
        ExperimentRunResult simResult;

        if (!effectiveExperimentName.empty()) {
            // Try to run the configured / CLI-selected experiment
            simResult =
                exec.runExperimentByName(module, effectiveExperimentName);
        } else {
            // No explicit or configured name -> first experiment
            simResult = exec.runFirstExperiment(module);
        }

        if (diag.hasErrors()) {
            diag.printToStdErr();
            return 1;
        }

        std::cout << "Simulated experiment '" << simResult.experimentName
                  << "':\n";
        for (const auto &kv : simResult.metrics) {
            std::cout << "  " << kv.first << " = " << kv.second << "\n";
        }
        return 0;
    }

    if (emitPytorchFlag || emitJaxFlag) {
        BackendRuntime backendRuntime(diag);

        BackendRuntimeOptions opts;
        opts.experimentName = effectiveExperimentName;
        opts.backendKind =
            emitJaxFlag ? RuntimeBackendKind::Jax : RuntimeBackendKind::Pytorch;

        std::string script = backendRuntime.emitScript(module, opts);
        if (script.empty() && diag.hasErrors()) {
            diag.printToStdErr();
            return 1;
        }

        std::cout << script;
        return 0;
    }

    if (dumpIRFlag) {
        dumpIR(module);
    } else {
        std::cout << "Parsed + analyzed + lowered to IR successfully:\n"
                  << "  Datasets   : " << module.datasets.size() << "\n"
                  << "  Models     : " << module.models.size() << "\n"
                  << "  Experiments: " << module.experiments.size() << "\n";
    }

    return 0;
}
