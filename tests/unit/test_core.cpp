#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "syntra/diagnostics.hpp"
#include "syntra/lexer.hpp"
#include "syntra/token.hpp"
#include "syntra/parser.hpp"
#include "syntra/ast.hpp"
#include "syntra/sema.hpp"
#include "syntra/ir.hpp"
#include "syntra/ir_pipeline.hpp"
#include "syntra/pass_manager.hpp"
#include "syntra/ir_execution.hpp"

using namespace syntra;

// ============================================================================
// Tiny test framework
// ============================================================================

struct TestCase {
    const char* name;
    void (*func)();
};

static std::vector<TestCase>& getRegistry() {
    static std::vector<TestCase> registry;
    return registry;
}

struct TestRegistrar {
    TestRegistrar(const char* name, void (*func)()) {
        getRegistry().push_back(TestCase{name, func});
    }
};

#define SYNTRA_TEST(name) \
    void name(); \
    static TestRegistrar registrar_##name(#name, &name); \
    void name()

#define SYNTRA_ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: " #cond " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while (0)

#define SYNTRA_ASSERT_EQ(expected, actual) \
    do { \
        auto _e = (expected); \
        auto _a = (actual); \
        if (!(_e == _a)) { \
            std::ostringstream _oss; \
            _oss << "Assertion failed: expected (" #expected ") == (" #actual ") " \
                 << "but they differ at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while (0)

// ============================================================================
// Helpers: run full front-end pipeline on a source string
// ============================================================================

struct FrontendResult {
    Program program;
    PipelineModule module;
};

static FrontendResult runFrontend(const std::string& virtualFilename,
                                  const std::string& source) {
    DiagnosticsEngine diag;

    Lexer lexer(virtualFilename, source, diag);
    auto tokens = lexer.tokenize();
    if (diag.hasErrors()) {
        diag.printToStdErr();
        throw std::runtime_error("lexer produced errors");
    }

    Parser parser(tokens, diag);
    Program program = parser.parseProgram();
    if (diag.hasErrors()) {
        diag.printToStdErr();
        throw std::runtime_error("parser produced errors");
    }

    SemanticAnalyzer sema(diag);
    sema.analyze(program);
    if (diag.hasErrors()) {
        diag.printToStdErr();
        throw std::runtime_error("semantic analyzer produced errors");
    }

    PipelineBuilder builder(diag);
    PipelineModule module = builder.build(program);
    if (diag.hasErrors()) {
        diag.printToStdErr();
        throw std::runtime_error("IR builder produced errors");
    }

    PassManager pm;
    pm.addDefaultPasses();
    pm.run(module, diag);
    if (diag.hasErrors()) {
        diag.printToStdErr();
        throw std::runtime_error("pass manager produced errors");
    }

    FrontendResult result;
    result.program = std::move(program);
    result.module  = std::move(module);
    return result;
}

// ============================================================================
// Test sources (internal mini Syntra programs)
// ============================================================================

static const char* BASIC_MNIST_SOURCE = R"syntra(
dataset mnist: ImageDataset {
    source  = "data/mnist_train.csv"
    shape   = (28, 28)
    channels = 1
    batch   = 64
    shuffle = true
    seed    = 42
}

model simple_cnn: TorchModel {
    arch      = "cnn"
    framework = "pytorch"
    lr        = 1e-3
    epochs    = 10
    optimizer = "adam"
}

experiment cnn_mnist(mnist) -> Metrics {
    let train_run = train(simple_cnn, mnist.train)

    let test_run = evaluate(simple_cnn, mnist.test) with {
        metrics = [accuracy, loss]
    }

    return {
        accuracy = test_run.accuracy,
        loss     = test_run.loss
    }
}
)syntra";

static const char* IMPLICIT_RETURN_SOURCE = R"syntra(
dataset mnist: ImageDataset {
    source  = "data/mnist_train.csv"
    batch   = 32
}

model simple_cnn: TorchModel {
    arch      = "cnn"
    framework = "pytorch"
    lr        = 1e-3
    epochs    = 5
    optimizer = "adam"
}

experiment cnn_mnist(mnist) -> Metrics {
    let train_run = train(simple_cnn, mnist.train)

    let test_run = evaluate(simple_cnn, mnist.test) with {
        metrics = [accuracy, loss]
    }
}
)syntra";

static const char* MULTI_EXP_SOURCE = R"syntra(
dataset mnist: ImageDataset {
    source  = "data/mnist_train.csv"
    batch   = 32
}

model simple_cnn: TorchModel {
    arch      = "cnn"
    framework = "pytorch"
    lr        = 1e-3
    epochs    = 5
    optimizer = "adam"
}

experiment exp_one(mnist) -> Metrics {
    let train_run = train(simple_cnn, mnist.train)
    let test_run  = evaluate(simple_cnn, mnist.test) with {
        metrics = [accuracy, loss]
    }
    return {
        accuracy = test_run.accuracy,
        loss     = test_run.loss
    }
}

experiment exp_two(mnist) -> Metrics {
    let train_run = train(simple_cnn, mnist.train)
    let test_run  = evaluate(simple_cnn, mnist.test) with {
        metrics = [accuracy, loss]
    }
    return {
        accuracy = test_run.accuracy,
        loss     = test_run.loss
    }
}
)syntra";

// ============================================================================
// Tests
// ============================================================================

SYNTRA_TEST(LexerProducesTokens) {
    DiagnosticsEngine diag;
    std::string source = BASIC_MNIST_SOURCE;

    Lexer lexer("test.syntra", source, diag);
    auto tokens = lexer.tokenize();

    SYNTRA_ASSERT_TRUE(!tokens.empty());

    // First token should be the 'dataset' keyword (whatever its TokenKind is),
    // but we can reliably check the lexeme.
    SYNTRA_ASSERT_EQ(std::string("dataset"), tokens[0].lexeme);

    SYNTRA_ASSERT_TRUE(!diag.hasErrors());
}

SYNTRA_TEST(ParserBuildsAST) {
    DiagnosticsEngine diag;
    std::string source = BASIC_MNIST_SOURCE;

    Lexer lexer("test.syntra", source, diag);
    auto tokens = lexer.tokenize();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    Parser parser(tokens, diag);
    Program program = parser.parseProgram();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    SYNTRA_ASSERT_EQ(1u, program.datasets.size());
    SYNTRA_ASSERT_EQ(1u, program.models.size());
    SYNTRA_ASSERT_EQ(1u, program.experiments.size());

    SYNTRA_ASSERT_EQ(std::string("mnist"), program.datasets[0].name);
    SYNTRA_ASSERT_EQ(std::string("simple_cnn"), program.models[0].name);
    SYNTRA_ASSERT_EQ(std::string("cnn_mnist"), program.experiments[0].name);
}

SYNTRA_TEST(SemaValidatesTypes) {
    DiagnosticsEngine diag;
    std::string source = BASIC_MNIST_SOURCE;

    Lexer lexer("test.syntra", source, diag);
    auto tokens = lexer.tokenize();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    Parser parser(tokens, diag);
    Program program = parser.parseProgram();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    SemanticAnalyzer sema(diag);
    sema.analyze(program);

    // Should be type-correct
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());
}

SYNTRA_TEST(IRBuilderCreatesPipeline) {
    std::string source = BASIC_MNIST_SOURCE;
    FrontendResult fe = runFrontend("basic_mnist.syntra", source);

    const PipelineModule& module = fe.module;

    SYNTRA_ASSERT_EQ(1u, module.datasets.size());
    SYNTRA_ASSERT_EQ(1u, module.models.size());
    SYNTRA_ASSERT_EQ(1u, module.experiments.size());

    const auto& exp = module.experiments[0];
    SYNTRA_ASSERT_EQ(std::string("cnn_mnist"), exp.name);

    // Expect exactly 2 ops: train + evaluate
    SYNTRA_ASSERT_EQ(2u, exp.ops.size());
    SYNTRA_ASSERT_TRUE(exp.ops[0].kind == PipelineOpKind::Train);
    SYNTRA_ASSERT_TRUE(exp.ops[1].kind == PipelineOpKind::Evaluate);

    // Explicit return mapping should have 2 fields
    SYNTRA_ASSERT_EQ(2u, exp.results.size());
    SYNTRA_ASSERT_EQ(std::string("accuracy"), exp.results[0].fieldName);
    SYNTRA_ASSERT_EQ(std::string("loss"), exp.results[1].fieldName);
}

SYNTRA_TEST(CanonicalizeResultsForImplicitReturn) {
    std::string source = IMPLICIT_RETURN_SOURCE;
    FrontendResult fe = runFrontend("implicit_return.syntra", source);

    const PipelineModule& module = fe.module;
    SYNTRA_ASSERT_EQ(1u, module.experiments.size());

    const auto& exp = module.experiments[0];

    // We didn't write "return {...}" in the source, so CanonicalizeResultsPass
    // should have synthesized a mapping from the last eval op's metrics.

    // Expect exactly 2 ops: train + evaluate
    SYNTRA_ASSERT_EQ(2u, exp.ops.size());
    SYNTRA_ASSERT_TRUE(exp.ops[0].kind == PipelineOpKind::Train);
    SYNTRA_ASSERT_TRUE(exp.ops[1].kind == PipelineOpKind::Evaluate);

    // We expect 2 result fields: accuracy + loss
    SYNTRA_ASSERT_EQ(2u, exp.results.size());

    bool seenAcc = false;
    bool seenLoss = false;
    for (const auto& rf : exp.results) {
        if (rf.fieldName == "accuracy") seenAcc = true;
        if (rf.fieldName == "loss")     seenLoss = true;
        // Both should come from the same eval var
        SYNTRA_ASSERT_EQ(std::string("test_run"), rf.sourceVar);
    }
    SYNTRA_ASSERT_TRUE(seenAcc);
    SYNTRA_ASSERT_TRUE(seenLoss);
}

SYNTRA_TEST(SimulatedExecutionProducesMetrics) {
    std::string source = BASIC_MNIST_SOURCE;
    FrontendResult fe = runFrontend("basic_mnist.syntra", source);

    DiagnosticsEngine diag;
    PipelineExecutor exec(diag);
    ExperimentRunResult res = exec.runFirstExperiment(fe.module);

    SYNTRA_ASSERT_EQ(std::string("cnn_mnist"), res.experimentName);
    SYNTRA_ASSERT_TRUE(!res.metrics.empty());

    // At least accuracy and loss should be present
    auto itAcc = res.metrics.find("accuracy");
    auto itLoss = res.metrics.find("loss");
    SYNTRA_ASSERT_TRUE(itAcc != res.metrics.end());
    SYNTRA_ASSERT_TRUE(itLoss != res.metrics.end());

    double acc = itAcc->second;
    double loss = itLoss->second;
    SYNTRA_ASSERT_TRUE(acc >= 0.0 && acc <= 1.0);
    SYNTRA_ASSERT_TRUE(loss >= 0.0 && loss <= 3.0);
}

SYNTRA_TEST(MultiExperimentExecutionByName) {
    // Run frontend pipeline
    DiagnosticsEngine diag;
    Lexer lexer("multi.syntra", MULTI_EXP_SOURCE, diag);
    auto tokens = lexer.tokenize();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    Parser parser(tokens, diag);
    Program program = parser.parseProgram();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    SemanticAnalyzer sema(diag);
    sema.analyze(program);
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    PipelineBuilder builder(diag);
    PipelineModule module = builder.build(program);
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    PassManager pm;
    pm.addDefaultPasses();
    pm.run(module, diag);
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    SYNTRA_ASSERT_EQ(2u, module.experiments.size());

    // Use in-process executor to run a specific experiment by name
    PipelineExecutor exec(diag);
    ExperimentRunResult res = exec.runExperimentByName(module, "exp_two");
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    SYNTRA_ASSERT_EQ(std::string("exp_two"), res.experimentName);

    // Basic sanity on metrics
    auto itAcc = res.metrics.find("accuracy");
    auto itLoss = res.metrics.find("loss");
    SYNTRA_ASSERT_TRUE(itAcc != res.metrics.end());
    SYNTRA_ASSERT_TRUE(itLoss != res.metrics.end());
}

SYNTRA_TEST(DumpASTProducesNonEmptyOutput) {
    DiagnosticsEngine diag;
    Lexer lexer("ast_test.syntra", BASIC_MNIST_SOURCE, diag);
    auto tokens = lexer.tokenize();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    Parser parser(tokens, diag);
    Program program = parser.parseProgram();
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    SemanticAnalyzer sema(diag);
    sema.analyze(program);
    SYNTRA_ASSERT_TRUE(!diag.hasErrors());

    std::string dump = astToString(program, /*withLocations=*/true);
    SYNTRA_ASSERT_TRUE(!dump.empty());

    // smoke check: should mention at least one of our top-level constructs
    SYNTRA_ASSERT_TRUE(dump.find("dataset") != std::string::npos
                       || dump.find("model") != std::string::npos
                       || dump.find("experiment") != std::string::npos);
}

// ============================================================================
// Test runner main
// ============================================================================

int main() {
    const auto& registry = getRegistry();
    std::size_t passed = 0;
    std::size_t failed = 0;

    std::cout << "Running " << registry.size() << " SyntraLine++ tests...\n";

    for (const auto& tc : registry) {
        std::cout << "[ RUN      ] " << tc.name << "\n";
        try {
            tc.func();
            std::cout << "[       OK ] " << tc.name << "\n";
            ++passed;
        } catch (const std::exception& ex) {
            std::cout << "[  FAILED  ] " << tc.name << "  --  " << ex.what() << "\n";
            ++failed;
        } catch (...) {
            std::cout << "[  FAILED  ] " << tc.name << "  --  unknown exception\n";
            ++failed;
        }
    }

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";

    return failed == 0 ? 0 : 1;
}