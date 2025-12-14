#pragma once

#include <string>
#include <unordered_map>
#include <optional>

#include "syntra/diagnostics.hpp"
#include "syntra/types.hpp"

namespace syntra {

enum class SymbolKind {
    Dataset,
    Model,
    Experiment,
};

struct Symbol {
    SymbolKind kind;
    std::string name;
    std::string typeName; // e.g. "ImageDataset", "TorchModel", "Metrics"
    SourceLocation loc;

    DatasetKind datasetKind() const {
        return datasetKindFromTypeName(typeName);
    }

    ModelKind modelKind() const {
        return modelKindFromTypeName(typeName);
    }

    ResultKind resultKind() const {
        return resultKindFromTypeName(typeName);
    }
};

/// Very simple global symbol table for v0.1:
///  - one namespace for all names
///  - we still track symbol kind to differentiate datasets/models/experiments.
class SymbolTable {
public:
    /// Try to insert a new symbol. On duplicate, reports an error and returns false.
    bool insert(SymbolKind kind,
                const std::string& name,
                const std::string& typeName,
                const SourceLocation& loc,
                DiagnosticsEngine& diag) {
        auto it = symbols_.find(name);
        if (it != symbols_.end()) {
            const Symbol& prev = it->second;
            std::string msg = "redefinition of '" + name + "'";
            diag.error(loc, msg);

            std::string note = "previous definition was here";
            diag.info(prev.loc, note);
            return false;
        }

        Symbol sym;
        sym.kind = kind;
        sym.name = name;
        sym.typeName = typeName;
        sym.loc = loc;
        symbols_.emplace(name, std::move(sym));
        return true;
    }

    /// Lookup a symbol by name. Returns nullptr if not found.
    const Symbol* lookup(const std::string& name) const {
        auto it = symbols_.find(name);
        if (it == symbols_.end()) {
            return nullptr;
        }
        return &it->second;
    }

private:
    std::unordered_map<std::string, Symbol> symbols_;
};

} // namespace syntra
