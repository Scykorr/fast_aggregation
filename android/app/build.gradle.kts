plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Версия читается из src/fast_aggregation/__init__.py. Формула versionCode
// совпадает с fast_aggregation.version.android_version_code:
// major * 100_000_000 + minor * 1_000_000 + patch * 10_000 + номер сборки CI.
val fastaggVersion = Regex("""__version__\s*=\s*["']([^"']+)["']""")
    .find(file("../../src/fast_aggregation/__init__.py").readText())
    ?.groupValues
    ?.get(1)
    ?: error("В src/fast_aggregation/__init__.py нет __version__")
val fastaggBuild = System.getenv("GITHUB_RUN_NUMBER")?.toIntOrNull() ?: 0
val fastaggRelease = Regex("""^(\d+)\.(\d+)\.(\d+)""").find(fastaggVersion)
    ?: error("версия должна быть major.minor.patch")
val (fastaggMajor, fastaggMinor, fastaggPatch) = fastaggRelease.groupValues.drop(1).map { it.toInt() }
val fastaggVersionCode =
    fastaggMajor * 100_000_000 + fastaggMinor * 1_000_000 + fastaggPatch * 10_000 + fastaggBuild
val fastaggVersionName = if (fastaggBuild > 0) "$fastaggVersion+$fastaggBuild" else fastaggVersion

android {
    namespace = "io.fastagg.node"
    compileSdk = 34
    buildFeatures {
        buildConfig = true
    }

    defaultConfig {
        applicationId = "io.fastagg.node"
        minSdk = 26
        targetSdk = 34
        versionCode = fastaggVersionCode
        versionName = fastaggVersionName
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.journeyapps:zxing-android-embedded:4.3.0")
    testImplementation("junit:junit:4.13.2")
}
