#pragma once

#include <string>
#include "ustring.hpp"
#include "onnxortext.h"


// Retrieves a vector of strings if the input type is std::string.
// It is a copy of the input data and can be modified to compute the output.
void GetTensorMutableDataString(
	const OrtApi& api, Ort::CustomOpApi& ort, OrtKernelContext* context,
	const OrtValue* value, std::vector<std::string>& output);

void GetTensorMutableDataString(
	const OrtApi& api, Ort::CustomOpApi& ort, OrtKernelContext* context,
	const OrtValue* value, std::vector<ustring>& output);

void FillTensorDataString(
	const OrtApi& api, Ort::CustomOpApi& ort, OrtKernelContext* context,
	const std::vector<std::string>& value, OrtValue* output);

void FillTensorDataString(
	const OrtApi& api, Ort::CustomOpApi& ort, OrtKernelContext* context,
	const std::vector<ustring>& value, OrtValue* output);

