#include <filesystem>
#include "gtest/gtest.h"
#include "onnxortext.h"
#include "utils/string_utils.h"
#include "utils/string_common.h"
#include "dll_onnxortext.h"
#include "common_test_ortops.h"

static CustomOpOne op_1st;
static CustomOpTwo op_2nd;

TEST(ops, test_ort_case) {
	auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

	std::vector<TestValue> inputs(2);
	inputs[0].name = "input_1";
	inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	inputs[0].dims = { 3, 5 };
	inputs[0].values_float = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
							  6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
							  11.1f, 12.2f, 13.3f, 14.4f, 15.5f };
	inputs[1].name = "input_2";
	inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	inputs[1].dims = { 3, 5 };
	inputs[1].values_float = { 15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
							  10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
							  5.5f, 4.4f, 3.3f, 2.2f, 1.1f };

	std::vector<TestValue> outputs(1);
	outputs[0].name = "output";
	outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	outputs[0].dims = { 3, 5 };
	outputs[0].values_int32 = { 17, 17, 17, 17, 17,
							   17, 18, 18, 18, 17,
							   17, 17, 17, 17, 17 };

	std::filesystem::path model_path = __FILE__;
	model_path = model_path.parent_path();
	model_path /= "..";
	model_path /= "data";
	model_path /= "custom_op_test.onnx";

	AddExternalCustomOp(&op_1st);
	AddExternalCustomOp(&op_2nd);
	TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(ustring, tensor_operator) {
	OrtValue* tensor;
	OrtAllocator* allocator;

	const auto* api_base = OrtGetApiBase();
	const auto* api = api_base->GetApi(ORT_API_VERSION);
	api->GetAllocatorWithDefaultOptions(&allocator);
	Ort::CustomOpApi custom_api(*api);

	std::vector<int64_t> dim{ 2, 2 };
	api->CreateTensorAsOrtValue(allocator, dim.data(), dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &tensor);

	std::vector<ustring> input_value{ ustring("test"), ustring("测试"), ustring("Test de"), ustring("🧐") };
	FillTensorDataString(*api, custom_api, nullptr, input_value, tensor);

	std::vector<ustring> output_value;
	GetTensorMutableDataString(*api, custom_api, nullptr, tensor, output_value);

	EXPECT_EQ(input_value, output_value);
}
