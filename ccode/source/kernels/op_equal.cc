#include <string>
#include "op_equal.hpp"
#include "op_equal_impl.hpp"

KernelStringEqual::KernelStringEqual(OrtApi api) : BaseKernel(api) {
}

void KernelStringEqual::Compute(OrtKernelContext* context) {
	KernelEqual_Compute<std::string>(api_, ort_, context);
}

size_t CustomOpStringEqual::GetInputTypeCount() const {
	return 2;
};

size_t CustomOpStringEqual::GetOutputTypeCount() const {
	return 1;
};

ONNXTensorElementDataType CustomOpStringEqual::GetOutputType(size_t /*index*/) const {
	return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
};

void* CustomOpStringEqual::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
	return new KernelStringEqual(api);
};

const char* CustomOpStringEqual::GetName() const {
	return "StringEqual";
};

ONNXTensorElementDataType CustomOpStringEqual::GetInputType(size_t /*index*/) const {
	return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
