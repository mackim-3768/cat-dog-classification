# Cats vs Dogs Classification Android App

This is a simple Android application that uses ExecuTorch to classify images as either a Cat or a Dog using a MobileNetV2 model.

## Prerequisites

1.  **Android Studio**: You need Android Studio installed to build and run this project.
2.  **ExecuTorch Libraries**: You need to build the ExecuTorch Android libraries (`libexecutorch.a`, etc.) and include them in the project.
3.  **Model File**: You need the exported `.pte` model file.

## Setup Instructions

### 1. Model File
Copy your exported `catsdogs_mobilenetv2.pte` file into the assets folder:
`app/src/main/assets/catsdogs_mobilenetv2.pte`

### 2. ExecuTorch Libraries
You need to integrate the ExecuTorch C++ libraries.
1.  Build ExecuTorch for Android (arm64-v8a).
2.  Update `app/src/main/cpp/CMakeLists.txt` to point to the include directories and link against the static libraries (`libexecutorch.a`, `libextension_module_static.a`, etc.).
3.  Update `app/src/main/cpp/native-lib.cpp` to include the actual headers and implement the `load` and `predict` functions using the ExecuTorch C++ API.

### 3. Build and Run
1.  Open the `android-app` folder in Android Studio.
2.  Sync Gradle.
3.  Connect your Android device (Exynos 2500 based).
4.  Run the app.

## Project Structure
-   `app/src/main/java/com/example/catsdogs/MainActivity.kt`: Handles Camera and UI.
-   `app/src/main/java/com/example/catsdogs/ModelRunner.kt`: Kotlin wrapper for the JNI calls.
-   `app/src/main/cpp/native-lib.cpp`: C++ JNI implementation (needs ExecuTorch integration).
-   `app/src/main/cpp/CMakeLists.txt`: CMake build configuration.
