#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// NOTE: You would include ExecuTorch headers here
// #include <executorch/extension/module/module.h>
// #include <executorch/extension/android/android_loader.h>

#define TAG "CatsDogsNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Placeholder for the actual ExecuTorch Module
// using namespace executorch::extension;
class ModelWrapper {
public:
    bool load(AAssetManager* assetManager, const std::string& assetName) {
        LOGI("Loading model %s...", assetName.c_str());
        // implementation would involve:
        // 1. Loading the .pte file from assets into a buffer or file descriptor
        // 2. Creating an ExecuTorch Module
        // module_ = Module::load(filePath);
        return true;
    }

    int predict(float* imageData, int length) {
        LOGI("Running prediction on image data of length %d", length);
        // implementation would involve:
        // 1. Preparing the inputs (tensor wrapping imageData)
        // 2. module_->forward(inputs)
        // 3. Parsing output tensor to find the index with max probability

        // Mock result for demo purposes if library is missing
        // In a real implementation, you'd run the model.
        return 1; // 1 for Dog, 0 for Cat (example)
    }
};

static ModelWrapper* gModel = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_catsdogs_ModelRunner_loadModel(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager,
        jstring assetName) {

    if (gModel) {
        delete gModel;
    }
    gModel = new ModelWrapper();

    const char* assetNameC = env->GetStringUTFChars(assetName, nullptr);
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    bool result = gModel->load(mgr, std::string(assetNameC));

    env->ReleaseStringUTFChars(assetName, assetNameC);
    return (jboolean)result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_catsdogs_ModelRunner_classifyImage(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray pixelData) {

    if (!gModel) {
        LOGE("Model not loaded!");
        return -1;
    }

    jsize len = env->GetArrayLength(pixelData);
    jfloat* data = env->GetFloatArrayElements(pixelData, nullptr);

    int result = gModel->predict(data, len);

    env->ReleaseFloatArrayElements(pixelData, data, 0);
    return result;
}
