#include <set>
#include "utils/string_utils.h"
#include "kernels/op_equal.hpp"
#include "kernels/string_split.hpp"
#include "kernels/debug_op.hpp"
#include "dll_onnxortext.h"

// A helper API to support test kernels.
// Must be invoked before RegisterCustomOps.

const char c_OpDomain[] = "ai.onnx.ext";

CustomOpStringEqual c_CustomOpStringEqual;
CustomOpStringSplit c_CustomOpStringSplit;

CustomOpOne op_1st;
CustomOpTwo op_2nd;

OrtCustomOp* operator_lists[] = {
	&op_1st,
	&op_2nd,
	&c_CustomOpStringEqual,
	&c_CustomOpStringSplit,
	nullptr };

extern "C" OrtStatus * ORT_API_CALL RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase * api) {
	OrtCustomOpDomain* domain = nullptr;
	const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);
	std::set<std::string> op_nameset;

	if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
		return status;
	}

	OrtCustomOp** ops = operator_lists;
	while (*ops != nullptr) {
		if (op_nameset.find((*ops)->GetName(*ops)) == op_nameset.end()) {
			if (auto status = ortApi->CustomOpDomain_Add(domain, *ops)) {
				return status;
			}
		}
		++ops;
	}

	return ortApi->AddCustomOpDomain(options, domain);
}

