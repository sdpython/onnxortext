#include "test_kernel.hpp"
#include <filesystem>
#include "gtest/gtest.h"
#include "onnxruntime_cxx_api.h"
#include "onnxortext.h"
#include "utils/string_utils.h"
#include "utils/string_common.h"

const char* GetLibraryPath();

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

template <typename T>
void _emplace_back(Ort::MemoryInfo& memory_info, std::vector<Ort::Value>& ort_inputs, const std::vector<T>& values, const std::vector<int64_t>& dims) {
	ort_inputs.emplace_back(Ort::Value::CreateTensor<T>(
		memory_info, const_cast<T*>(values.data()), values.size(), dims.data(), dims.size()));
}

template <typename T>
void _assert_eq(Ort::Value& output_tensor, const std::vector<T>& expected, size_t total_len) {
	ASSERT_EQ(expected.size(), total_len);
	T* f = output_tensor.GetTensorMutableData<T>();
	for (size_t i = 0; i != total_len; ++i) {
		ASSERT_EQ(expected[i], f[i]);
	}
}

void GetTensorMutableDataString(
	const OrtApi& api,
	const OrtValue* value,
	std::vector<std::string>& output);

void RunSession(Ort::Session& session_object,
	const std::vector<TestValue>& inputs,
	const std::vector<TestValue>& outputs);

void TestInference(Ort::Env& env, const ORTCHAR_T* model_uri,
	const std::vector<TestValue>& inputs,
	const std::vector<TestValue>& outputs,
	const char* custom_op_library_filename);

