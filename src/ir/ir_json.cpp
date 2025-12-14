#include "syntra/ir_json.hpp"

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "syntra/ir.hpp"

namespace syntra {

// -----------------------------------------------------------------------------
// Minimal JSON writer (no external dependencies)
// -----------------------------------------------------------------------------

namespace {

struct JsonWriter {
    std::ostringstream out;
    int indent = 0;
    bool pretty = true;
    bool firstInContainer = true;

    explicit JsonWriter(bool prettyFormatting) : pretty(prettyFormatting) {}

    void writeIndent() {
        if (!pretty)
            return;
        for (int i = 0; i < indent; ++i) {
            out << ' ';
        }
    }

    void newline() {
        if (pretty) {
            out << '\n';
        }
    }

    // Escapes " and \ minimally; our IR strings are simple identifiers,
    // so we don't need full JSON escaping for now.
    static void writeEscapedString(std::ostringstream &os,
                                   const std::string &s) {
        os << '"';
        for (char c : s) {
            switch (c) {
            case '\\':
                os << "\\\\";
                break;
            case '"':
                os << "\\\"";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                os << c;
                break;
            }
        }
        os << '"';
    }

    void beginObject() {
        out << "{";
        indent += pretty ? 2 : 0;
        firstInContainer = true;
        newline();
    }

    void endObject() {
        indent -= pretty ? 2 : 0;
        newline();
        writeIndent();
        out << "}";
        firstInContainer = false;
    }

    void beginArray() {
        out << "[";
        indent += pretty ? 2 : 0;
        firstInContainer = true;
        newline();
    }

    void endArray() {
        indent -= pretty ? 2 : 0;
        newline();
        writeIndent();
        out << "]";
        firstInContainer = false;
    }

    void commaNewlineIfNeeded() {
        if (!firstInContainer) {
            out << ",";
            newline();
        }
        firstInContainer = false;
    }

    // "key": <value>
    template <typename F> void keyValue(const std::string &key, F valueWriter) {
        commaNewlineIfNeeded();
        writeIndent();
        writeEscapedString(out, key);
        out << (pretty ? ": " : ":");
        valueWriter();
    }

    void writeString(const std::string &value) {
        writeEscapedString(out, value);
    }

    void writeArrayOfStrings(const std::vector<std::string> &values) {
        out << "[";
        if (!values.empty() && pretty) {
            newline();
            indent += 2;
        } else if (!values.empty()) {
            out << ' ';
        }

        bool first = true;
        for (const auto &v : values) {
            if (!first) {
                out << ",";
                if (pretty)
                    newline();
                else
                    out << ' ';
            }
            first = false;
            if (pretty)
                writeIndent();
            writeEscapedString(out, v);
        }

        if (!values.empty() && pretty) {
            indent -= 2;
            newline();
            writeIndent();
        } else if (!values.empty()) {
            out << ' ';
        }

        out << "]";
    }

}; // struct JsonWriter

} // namespace

// -----------------------------------------------------------------------------
// Serialization helpers
// -----------------------------------------------------------------------------

std::string pipelineToJson(const PipelineModule &module, bool pretty) {
    JsonWriter w(pretty);

    // Root object
    w.beginObject();

    // -------------------------------------------------------------------------
    // datasets
    // -------------------------------------------------------------------------
    w.keyValue("datasets", [&]() {
        w.beginArray();

        bool firstDataset = true;
        for (const auto &d : module.datasets) {
            if (!firstDataset) {
                w.out << ",";
                w.newline();
            }
            firstDataset = false;

            w.writeIndent();
            w.beginObject();

            w.keyValue("name", [&]() { w.writeString(d.name); });
            w.keyValue("type", [&]() { w.writeString(d.typeName); });

            // In the future, we can add serialized field configs here
            // once the IR structure stabilizes for dataset fields.

            w.endObject();
        }

        w.endArray();
    });

    // -------------------------------------------------------------------------
    // models
    // -------------------------------------------------------------------------
    w.keyValue("models", [&]() {
        w.beginArray();

        bool firstModel = true;
        for (const auto &m : module.models) {
            if (!firstModel) {
                w.out << ",";
                w.newline();
            }
            firstModel = false;

            w.writeIndent();
            w.beginObject();

            w.keyValue("name", [&]() { w.writeString(m.name); });
            w.keyValue("type", [&]() { w.writeString(m.typeName); });

            // Similarly, we could add model hyperparameters here later.

            w.endObject();
        }

        w.endArray();
    });

    // -------------------------------------------------------------------------
    // experiments
    // -------------------------------------------------------------------------
    w.keyValue("experiments", [&]() {
        w.beginArray();

        bool firstExperiment = true;
        for (const auto &e : module.experiments) {
            if (!firstExperiment) {
                w.out << ",";
                w.newline();
            }
            firstExperiment = false;

            w.writeIndent();
            w.beginObject();

            w.keyValue("name", [&]() { w.writeString(e.name); });
            w.keyValue("dataset_param",
                       [&]() { w.writeString(e.datasetParam); });
            w.keyValue("result_type",
                       [&]() { w.writeString(e.resultTypeName); });

            // -----------------------------
            // ops
            // -----------------------------
            w.keyValue("ops", [&]() {
                w.beginArray();

                bool firstOp = true;
                for (const auto &op : e.ops) {
                    if (!firstOp) {
                        w.out << ",";
                        w.newline();
                    }
                    firstOp = false;

                    w.writeIndent();
                    w.beginObject();

                    switch (op.kind) {
                    case PipelineOpKind::Train:
                        w.keyValue("kind", [&]() { w.writeString("train"); });
                        w.keyValue("var",
                                   [&]() { w.writeString(op.train.varName); });
                        w.keyValue("model", [&]() {
                            w.writeString(op.train.modelName);
                        });
                        w.keyValue("dataset", [&]() {
                            w.writeString(op.train.datasetName);
                        });
                        w.keyValue("split", [&]() {
                            w.writeString(op.train.splitName);
                        });
                        break;

                    case PipelineOpKind::Evaluate:
                        w.keyValue("kind",
                                   [&]() { w.writeString("evaluate"); });
                        w.keyValue("var",
                                   [&]() { w.writeString(op.eval.varName); });
                        w.keyValue("model",
                                   [&]() { w.writeString(op.eval.modelName); });
                        w.keyValue("dataset", [&]() {
                            w.writeString(op.eval.datasetName);
                        });
                        w.keyValue("split",
                                   [&]() { w.writeString(op.eval.splitName); });
                        w.keyValue("metrics", [&]() {
                            w.writeArrayOfStrings(op.eval.metrics);
                        });
                        break;
                    }

                    w.endObject();
                }

                w.endArray();
            });

            // -----------------------------
            // results (return {...})
            // -----------------------------
            w.keyValue("results", [&]() {
                w.beginArray();

                bool firstRes = true;
                for (const auto &rf : e.results) {
                    if (!firstRes) {
                        w.out << ",";
                        w.newline();
                    }
                    firstRes = false;

                    w.writeIndent();
                    w.beginObject();

                    w.keyValue("field", [&]() { w.writeString(rf.fieldName); });
                    w.keyValue("source_var",
                               [&]() { w.writeString(rf.sourceVar); });
                    w.keyValue("metric",
                               [&]() { w.writeString(rf.metricName); });

                    w.endObject();
                }

                w.endArray();
            });

            w.endObject();
        }

        w.endArray();
    });

    // End root object
    w.endObject();
    w.newline();

    return w.out.str();
}

} // namespace syntra
