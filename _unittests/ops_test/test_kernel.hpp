#pragma once

#include <string>
#include <vector>
#include <math.h>
#include "onnxortext.h"

const char* GetLibraryPath();

struct TestValue {
	const char* name = nullptr;
	ONNXTensorElementDataType element_type;
	std::vector<int64_t> dims;
	std::vector<float> values_float;
	std::vector<uint8_t> values_uint8;
	std::vector<int32_t> values_int32;
	std::vector<int64_t> values_int64;
	std::vector<std::string> values_string;
};

void RunSession(Ort::Session& session_object,
	const std::vector<TestValue>& inputs,
	const std::vector<TestValue>& outputs);

void TestInference(const ORTCHAR_T* model_uri,
	const std::vector<TestValue>& inputs,
	const std::vector<TestValue>& outputs,
	const char* custom_op_library_filename);
