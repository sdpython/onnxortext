#pragma once
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

	void ORT_API_CALL InitLib();
	OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
	bool ORT_API_CALL AddExternalCustomOp(const OrtCustomOp* c_op);

#ifdef __cplusplus
}
#endif
