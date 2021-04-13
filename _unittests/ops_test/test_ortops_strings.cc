#include <filesystem>
#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "test_kernel.hpp"
#include "common_test_ortops.h"

TEST(ops, test_string_split) {
	std::vector<TestValue> inputs(3);
	inputs[0].name = "input";
	inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	inputs[0].dims = { 3 };
	inputs[0].values_string = { "Abc cc", "Abcé éé", "中文 文" };

	inputs[1].name = "delimiter";
	inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	inputs[1].dims = { 1 };
	inputs[1].values_string = { " " };

	inputs[2].name = "skip_empty";
	inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
	inputs[2].dims = { 1 };
	inputs[2].values_uint8 = { 0 };

	std::vector<TestValue> outputs(3);
	outputs[0].name = "indices";
	outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
	outputs[0].dims = { 6, 2 };
	outputs[0].values_int64 = {
		0, 0, 0, 1, 
		1, 0, 1, 1, 
		2, 0, 2, 1 };

	outputs[1].name = "values";
	outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	outputs[1].dims = { 6 };
	outputs[1].values_string = { 
		"Abc", "cc", "Abcé", "éé", "中文", "文" };

	outputs[2].name = "shape";
	outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
	outputs[2].dims = { 2 };
	outputs[2].values_int64 = { 3, 2 };

	std::filesystem::path model_path = __FILE__;
	model_path = model_path.parent_path();
	model_path /= "..";
	model_path /= "data";
	model_path /= "custom_op_string_split.onnx";
	TestInference(model_path.c_str(), inputs, outputs, GetLibraryPath());
}
