#pragma once

#include "utils/string_utils.h"
#include "onnxortext.h"

struct KernelOne : BaseKernel {
	KernelOne(OrtApi api);
	void Compute(OrtKernelContext* context);
};

struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
	const char* GetName() const;
	size_t GetInputTypeCount() const;
	ONNXTensorElementDataType GetInputType(size_t index) const;
	size_t GetOutputTypeCount() const;
	ONNXTensorElementDataType GetOutputType(size_t index) const;
};

struct KernelTwo : BaseKernel {
	KernelTwo(OrtApi api);
	void Compute(OrtKernelContext* context);
};

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
	const char* GetName() const;
	size_t GetInputTypeCount() const;
	ONNXTensorElementDataType GetInputType(size_t index) const;
	size_t GetOutputTypeCount() const;
	ONNXTensorElementDataType GetOutputType(size_t index) const;
};
