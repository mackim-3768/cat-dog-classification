# Issue: Finalize ExecuTorch Integration

**Status**: Open
**Related Branch**: jules-develop (or current working branch)
**Description**: The current Android application uses a "Mock" implementation for the JNI layer to ensure the project structure compiles and runs in the absence of the ExecuTorch build environment.

The following tasks must be completed to enable actual on-device inference:

## 1. Build ExecuTorch Libraries
**Action**: Build the ExecuTorch runtime for Android.
**Details**:
- Target ABI: `arm64-v8a`
- Required Libraries: `libexecutorch.a`, `libextension_module_static.a` (and any other backend-specific libs like `libbackend_samsung.a` if using E9955 specifically).

## 2. Update CMake Configuration
**File**: `android-app/app/src/main/cpp/CMakeLists.txt`
**Tasks**:
- Define the imported static libraries for ExecuTorch.
- Link them to the `catsdogs` target.
- Example:
  ```cmake
  add_library(executorch STATIC IMPORTED)
  set_target_properties(executorch PROPERTIES IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../executorch/cmake-out/libexecutorch.a")
  target_link_libraries(catsdogs executorch ...)
  ```

## 3. Implement C++ JNI Layer
**File**: `android-app/app/src/main/cpp/native-lib.cpp`
**Tasks**:
- **Include Headers**: Uncomment `#include <executorch/...>`.
- **Implement `ModelWrapper::load`**:
  - Read the asset into a memory buffer (using `AAsset_getBuffer`).
  - Use `executorch::extension::Module::load()` to load the model from the buffer.
- **Implement `ModelWrapper::predict`**:
  - Wrap the incoming `float* imageData` (size 224*224*3) into an `executorch::Tensor`.
  - Execute `module_->forward()`.
  - Parse the output Tensor to find the class index with the highest score.

## 4. Add Model Asset
**File**: `android-app/app/src/main/assets/catsdogs_mobilenetv2.pte`
**Action**: Copy the exported model file to this location.
