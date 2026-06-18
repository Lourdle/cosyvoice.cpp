# Android 构建指南

## Termux（手机上直接编译）

### 前置条件

在 Android 手机上安装并打开 Termux，先安装必要的工具和库：

```bash
pkg update
apt install clang cmake ninja git simde
```

如需 FFmpeg 后端则加装 `apt install ffmpeg`，如需 ICU 则加装 `apt install libicu`。如需前端，则安装 `apt install onnxruntime`，CMake 会自动通过 `find_package` 找到系统版本。

### CPU

按照 README 中的一般流程配置和编译即可。前置条件中已通过 `apt install onnxruntime` 安装了 ONNX Runtime，CMake 会自动通过 `find_package` 找到系统版本。如需使用 ICU 则再加装 `apt install libicu`。

### OpenCL

如果要在 Termux 上启用 OpenCL，需要先准备 OpenCL 头文件：

```bash
git clone https://github.com/KhronosGroup/OpenCL-Headers.git --depth=1
```

然后确认系统里可访问 OpenCL 运行时。Android 的策略在不同版本里有变化：

- 早期 Android 可能可以直接使用 `/vendor/lib64/libOpenCL.so`
- Android 16 需要使用 `/system_ext/lib64/libOpenCL_system.so`
- 如果 `$PREFIX/lib/libOpenCL.so` 已经存在，且当前系统是 Android 16，建议把它改成指向 `libOpenCL_system.so` 的符号链接

```bash
rm -f $PREFIX/lib/libOpenCL.so
ln -s /system_ext/lib64/libOpenCL_system.so $PREFIX/lib/libOpenCL.so
```

> 这样做的目的是让链接阶段直接命中目标文件名 `libOpenCL_system.so`。Android 16 上 `ggml-opencl` 需要先通过符号链接指向这个名字，不能直接复制文件。

接着按正常方式配置，只是额外打开 OpenCL，并关闭 Adreno kernels：

```bash
cmake -S /path/to/cosyvoice.cpp -B build-termux-opencl \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INCLUDE_PATH=/path/to/OpenCL-Headers \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
ninja -C build-termux-opencl
```

> `GGML_OPENCL_USE_ADRENO_KERNELS` 即使目标 GPU 是 Adreno 也需要设为 `OFF`，因为内部优化的 kernel 对 tensor 形状有特定要求，CosyVoice 不满足这些形状约束。

`ldd` 输出里如果已经看到 `libOpenCL_system.so => not found`，说明 `ggml-opencl` 这边已经按目标文件名链接上了。

![OpenCL link target](assets/termux-opencl-link.jpg)

如果使用 FFmpeg 的音频后端，可能会出现 `CANNOT LINK EXECUTABLE "./cosyvoice-cli": cannot find "libOpenCL.so" from verneed[1] in DT_NEEDED list for "/data/data/com.termux/files/usr/lib/libavutil.so.60.26.101"`。这是因为 FFmpeg 需要使用原始的 `libOpenCL.so`。如果 `ldd libggml-opencl.so` 已经看到它链接的是 `libOpenCL_system.so`，并且你要使用 FFmpeg 音频后端，就可以重新安装 `ocl-icd`，把它恢复成 FFmpeg 可用的原始 `so`：

```bash
apt reinstall ocl-icd
```

图里就是正常进入交互模式后的状态。

![Interactive mode](assets/termux-opencl-interactive.jpg)

## 交叉编译（跨平台编译）

### 前置条件

**Android NDK**

从 [https://developer.android.google.cn/ndk/downloads](https://developer.android.google.cn/ndk/downloads) 下载 NDK（toolchain 需要支持 C++20），解压到任意目录。

**SIMDe**

```bash
git clone https://github.com/simd-everywhere/simde.git --depth=1
```

**CMake（≥ 3.24）和 Ninja**

安装后确保 `cmake` 和 `ninja` 可用。

### CPU

```bash
cmake -S /path/to/cosyvoice.cpp -B build-android-cpu \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-rXX/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_PLATFORM=24 \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_STL=c++_shared \
  -DSIMDE_INCLUDE_DIR=/path/to/simde \
  -DCOSYVOICE_NO_FRONTEND=ON \
  -DCOSYVOICE_NO_ICU=ON \
  -G Ninja
```

| 参数 | 说明 |
|---|---|
| `CMAKE_TOOLCHAIN_FILE` | NDK 的 Android 工具链文件 |
| `CMAKE_BUILD_TYPE` | Ninja 为单配置生成器，需要指定 |
| `ANDROID_PLATFORM` | 最低 Android API 级别 |
| `ANDROID_ABI` | arm64-v8a |
| `ANDROID_STL` | C++ 运行时库 |
| `SIMDE_INCLUDE_DIR` | SIMDe 目录，需包含 `simde/x86/avx2.h` |
| `COSYVOICE_NO_FRONTEND` | ONNX Runtime 无 Android 预编译库，故关闭 |
| `COSYVOICE_NO_ICU` | ICU 无 Android 预编译库，故关闭 |

```bash
ninja -C build-android-cpu
cmake --build build-android-cpu --config Release -j8
```

### OpenCL

先准备 OpenCL Headers：

```bash
git clone https://github.com/KhronosGroup/OpenCL-Headers.git --depth=1
cp -r OpenCL-Headers/CL /path/to/ndk/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include/
```

> 将 `windows-x86_64` 替换为你所用主机的对应目录（例如 `linux-x86_64`、`darwin-x86_64`）。

再编译 OpenCL ICD Loader：

```bash
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git --depth=1
cd OpenCL-ICD-Loader
cmake -B build-ndk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=/path/to/ndk/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=24 \
  -DANDROID_STL=c++_shared
ninja -C build-ndk
cp build-ndk/libOpenCL.so /path/to/ndk/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/
```

最后编译 cosyvoice.cpp：

```bash
cmake -S /path/to/cosyvoice.cpp -B build-android-opencl \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_PLATFORM=24 \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_STL=c++_shared \
  -DSIMDE_INCLUDE_DIR=/path/to/simde \
  -DCOSYVOICE_NO_FRONTEND=ON \
  -DCOSYVOICE_NO_ICU=ON \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -G Ninja
ninja -C build-android-opencl
```

> `GGML_OPENCL_USE_ADRENO_KERNELS` 即使目标 GPU 是 Adreno 也需要设为 `OFF`，因为内部优化的 kernel 对 tensor 形状有特定要求，CosyVoice 不满足这些形状约束。
