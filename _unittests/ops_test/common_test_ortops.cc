#include "test_kernel.hpp"
#include <filesystem>
#include "gtest/gtest.h"
#include "onnxruntime_cxx_api.h"
#include "onnxortext.h"
#include "utils/string_utils.h"
#include "utils/string_common.h"
#include "common_test_ortops.h"

const char* GetLibraryPath() {
#if defined(_WIN32)
	return "onnxortext.dll";
#elif defined(__APPLE__)
	return "libonnxortext.dylib";
#else
	return "./libonnxortext.so";
#endif
}

void GetTensorMutableDataString(const OrtApi& api, const OrtValue* value, std::vector<std::string>& output) {
	Ort::CustomOpApi ort(api);
	OrtTensorDimensions dimensions(ort, value);
	size_t len = static_cast<size_t>(dimensions.Size());
	size_t data_len;
	Ort::ThrowOnError(api, api.GetStringTensorDataLength(value, &data_len));
	output.resize(len);
	std::vector<char> result(data_len + len + 1, '\0');
	std::vector<size_t> offsets(len);
	Ort::ThrowOnError(
		api, api.GetStringTensorContent(
			value, (void*)result.data(), data_len, offsets.data(), offsets.size()));
	output.resize(len);
	for (int64_t i = (int64_t)len - 1; i >= 0; --i) {
		if (i < len - 1)
			result[offsets[i + (int64_t)1]] = '\0';
		output[i] = result.data() + offsets[i];
	}
}

void RunSession(Ort::Session& session_object,
	const std::vector<TestValue>& inputs,
	const std::vector<TestValue>& outputs) {
	std::vector<Ort::Value> ort_inputs;
	std::vector<const char*> input_names;
	std::vector<const char*> output_names;

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::AllocatorWithDefaultOptions allocator;

	for (size_t i = 0; i < inputs.size(); i++) {
		input_names.emplace_back(inputs[i].name);
		switch (inputs[i].element_type) {
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			_emplace_back(memory_info, ort_inputs, inputs[i].values_float, inputs[i].dims);
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			_emplace_back(memory_info, ort_inputs, inputs[i].values_int32, inputs[i].dims);
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			_emplace_back(memory_info, ort_inputs, inputs[i].values_int64, inputs[i].dims);
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			_emplace_back(memory_info, ort_inputs, inputs[i].values_uint8, inputs[i].dims);
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
			Ort::Value& ort_value = ort_inputs.emplace_back(
				Ort::Value::CreateTensor(
					allocator, inputs[i].dims.data(), inputs[i].dims.size(),
					ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));
			for (size_t i_str = 0; i_str < inputs[i].values_string.size(); ++i_str) {
				ort_value.FillStringTensorElement(
					inputs[i].values_string[i_str].c_str(), i_str);
			}
		} break;
		default:
			throw std::runtime_error(MakeString(
				"Unable to handle input ", i, " type ", inputs[i].element_type,
				" is not implemented yet."));
		}
	}
	for (size_t index = 0; index < outputs.size(); ++index) {
		output_names.push_back(outputs[index].name);
	}

	std::vector<Ort::Value> ort_outputs;
	ort_outputs = session_object.Run(Ort::RunOptions{ nullptr },
		input_names.data(), ort_inputs.data(), ort_inputs.size(),
		output_names.data(), outputs.size());
	ASSERT_EQ(outputs.size(), ort_outputs.size());
	for (size_t index = 0; index < outputs.size(); ++index) {
		auto output_tensor = &ort_outputs[index];
		const TestValue& expected = outputs[index];

		auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType output_type = type_info.GetElementType();
		ASSERT_EQ(output_type, expected.element_type);
		std::vector<int64_t> dimension = type_info.GetShape();
		ASSERT_EQ(dimension, expected.dims);
		size_t total_len = type_info.GetElementCount();
		switch (expected.element_type) {
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			_assert_eq(*output_tensor, expected.values_float, total_len);
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			_assert_eq(*output_tensor, expected.values_int32, total_len);
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			_assert_eq(*output_tensor, expected.values_int64, total_len);
			break;
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
			std::vector<std::string> output_string;
			GetTensorMutableDataString(Ort::GetApi(), *output_tensor, output_string);
			ASSERT_EQ(expected.values_string, output_string);
			break;
		}
		default:
			throw std::runtime_error(MakeString(
				"Unable to handle output ", index, " type ", expected.element_type,
				" is not implemented yet."));
		}
	}
}

void TestInference(const ORTCHAR_T* model_uri,
	const std::vector<TestValue>& inputs,
	const std::vector<TestValue>& outputs,
	const char* custom_op_library_filename) {
	Ort::InitApi();
	auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

	Ort::SessionOptions session_options;
	void* handle = nullptr;
	if (custom_op_library_filename) {
		Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(
			(OrtSessionOptions*)session_options, custom_op_library_filename, &handle));
	}

	// if session creation passes, model loads fine
	Ort::Session session(*ort_env, model_uri, session_options);

	// Now run
	RunSession(session, inputs, outputs);
}
