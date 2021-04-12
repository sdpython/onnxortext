#include <set>
#include "utils/string_utils.h"
#include "kernels/op_equal.hpp"
#include "kernels/string_split.hpp"
#include "dll_onnxortext.h"

// A helper API to support test kernels.
// Must be invoked before RegisterCustomOps.

const char c_OpDomain[] = "ai.onnx.ext";



CustomOpStringEqual c_CustomOpStringEqual;
CustomOpStringSplit c_CustomOpStringSplit;

OrtCustomOp* operator_lists[] = {
    &c_CustomOpStringEqual,
    &c_CustomOpStringSplit,
    nullptr };

class ExternalCustomOps {
public:
    ExternalCustomOps() {
    }

    static ExternalCustomOps& instance() {
        static ExternalCustomOps g_instance;
        return g_instance;
    }

    void Add(const OrtCustomOp* c_op) {
        op_array_.push_back(c_op);
    }

    const OrtCustomOp* GetNextOp(size_t& idx) {
        if (idx >= op_array_.size()) {
            return nullptr;
        }

        return op_array_[idx++];
    }

    ExternalCustomOps(ExternalCustomOps const&) = delete;
    void operator=(ExternalCustomOps const&) = delete;

private:
    std::vector<const OrtCustomOp*> op_array_;
};

extern "C" bool AddExternalCustomOp(const OrtCustomOp * c_op) {
    ExternalCustomOps::instance().Add(c_op);
    return true;
}

extern "C" OrtStatus * ORT_API_CALL RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase * api) {
    OrtCustomOpDomain* domain = nullptr;
    const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);
    std::set<std::string> pyop_nameset;

    if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
        return status;
    }

    OrtCustomOp** ops = operator_lists;
    while (*ops != nullptr) {
        if (pyop_nameset.find((*ops)->GetName(*ops)) == pyop_nameset.end()) {
            if (auto status = ortApi->CustomOpDomain_Add(domain, *ops)) {
                return status;
            }
        }
        ++ops;
    }

    return ortApi->AddCustomOpDomain(options, domain);
}

