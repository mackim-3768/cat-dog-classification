package com.example.catsdogs

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log

class ModelRunner(private val context: Context) {

    companion object {
        init {
            System.loadLibrary("catsdogs")
        }
    }

    private external fun loadModel(assetManager: AssetManager, assetName: String): Boolean
    private external fun classifyImage(pixelData: FloatArray): Int

    fun load(modelName: String): Boolean {
        return loadModel(context.assets, modelName)
    }

    fun classify(bitmap: Bitmap): String {
        // Preprocess the image
        // 1. Resize to 224x224 (assuming MobileNetV2 input size)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // 2. Convert to FloatArray and Normalize
        val inputData = preprocess(resizedBitmap)

        // 3. Run inference
        val resultIndex = classifyImage(inputData)

        return when (resultIndex) {
            0 -> "Cat"
            1 -> "Dog"
            else -> "Unknown"
        }
    }

    private fun preprocess(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val floatArray = FloatArray(width * height * 3)
        val intValues = IntArray(width * height)

        bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

        var pixel = 0
        for (i in 0 until width * height) {
            val value = intValues[i]

            // Normalize assuming Mean=[0.485, 0.456, 0.406] and Std=[0.229, 0.224, 0.225]
            // Or simpler normalization [0, 1] -> (value / 255.0f)
            // Here we do simple (x / 255.0) for demo simplicity, but should match training

            floatArray[pixel++] = ((value shr 16 and 0xFF) / 255.0f - 0.485f) / 0.229f // R
            floatArray[pixel++] = ((value shr 8 and 0xFF) / 255.0f - 0.456f) / 0.224f  // G
            floatArray[pixel++] = ((value and 0xFF) / 255.0f - 0.406f) / 0.225f        // B
        }
        return floatArray
    }
}
