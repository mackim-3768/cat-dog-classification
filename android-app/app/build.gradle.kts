plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.catsdogs"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.catsdogs"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags("-std=c++17")
                // abiFilters "arm64-v8a" // Uncomment to target specific ABIs
            }
        }

        // Ensure assets are included
        sourceSets {
            getByName("main") {
                assets.srcDirs("src/main/assets")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // CameraX
    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:${cameraxVersion}")
    implementation("androidx.camera:camera-camera2:${cameraxVersion}")
    implementation("androidx.camera:camera-lifecycle:${cameraxVersion}")
    implementation("androidx.camera:camera-view:${cameraxVersion}")
    implementation("androidx.camera:camera-extensions:${cameraxVersion}")

    // ExecuTorch - The user is expected to add the local dependency or AAR here.
    // For example:
    // implementation(files("libs/executorch.aar"))
    // implementation(project(":executorch"))
}
