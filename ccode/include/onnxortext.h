#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

struct BaseKernel {
  BaseKernel(OrtApi api) : api_(api), info_(nullptr), ort_(api_) {}
  BaseKernel(OrtApi api, const OrtKernelInfo* info) : api_(api), info_(info), ort_(api_) {}

  bool HasAttribute(const char* name) const;
  template <class T>
  bool TryToGetAttribute(const char* name, T& value);

 protected:
  OrtErrorCode GetErrorCodeAndRelease(OrtStatusPtr status);
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  const OrtKernelInfo* info_;
};

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions() = default;
  OrtTensorDimensions(Ort::CustomOpApi& ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
  const std::vector<int64_t>& GetDims() const { return *this; }
  int64_t Size() const {
    int64_t s = 1.;
    for (auto it = begin(); it != end(); ++it)
      s *= *it;
    return s;
  }
};
