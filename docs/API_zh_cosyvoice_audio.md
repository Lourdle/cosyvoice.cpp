# cosyvoice-audio.h API 参考

本文档覆盖 `include/cosyvoice-audio.h` 中声明的全部公开符号。

## COSYVOICE_API

### 语法

```c
#ifndef COSYVOICE_API
    #ifdef COSYVOICE_STATIC
        #define COSYVOICE_API
    #else
        #ifdef _WIN32
            #define COSYVOICE_API __declspec(dllimport)
        #else
            #define COSYVOICE_API __attribute__((visibility("default")))
        #endif
    #endif
#endif
```

### 说明

控制音频工具函数的导入导出可见性。

## cosyvoice_audio_load_from_file

### 语法

```c
COSYVOICE_API bool cosyvoice_audio_load_from_file(
    const char* filename,
    float**     data,
    uint32_t*   length,
    uint32_t*   sample_rate
);
```

### 说明

加载音频文件并解码为单声道浮点 PCM。

### 参数

- `filename`：输入文件路径。
- `data`：输出采样缓冲区。
- `length`：输出采样点数量。
- `sample_rate`：输出采样率（Hz）。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

返回的 `data` 需要使用 `cosyvoice_audio_free` 释放。

## cosyvoice_audio_resample

### 语法

```c
COSYVOICE_API bool cosyvoice_audio_resample(
    const float* input,
    uint32_t     input_length,
    uint32_t     input_sample_rate,
    float**      output,
    uint32_t*    output_length,
    uint32_t     output_sample_rate
);
```

### 说明

将单声道浮点 PCM 重采样到目标采样率。

### 参数

- `input`：输入采样数据。
- `input_length`：输入采样点数量。
- `input_sample_rate`：输入采样率（Hz）。
- `output`：输出重采样缓冲区。
- `output_length`：输出采样点数量。
- `output_sample_rate`：目标采样率（Hz）。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

返回的 `output` 需要使用 `cosyvoice_audio_free` 释放。

## cosyvoice_audio_save_to_file

### 语法

```c
COSYVOICE_API bool cosyvoice_audio_save_to_file(
    const char*  filename,
    const float* data,
    uint32_t     length,
    uint32_t     sample_rate
);
```

### 说明

将单声道浮点 PCM 编码并保存到音频文件。

### 参数

- `filename`：输出文件路径。
- `data`：输入采样数据。
- `length`：采样点数量。
- `sample_rate`：输出采样率（Hz）。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

编码格式由输出文件扩展名决定：

- `.wav` -> WAV
- `.flac` -> FLAC
- `.mp3` -> MP3
- `.ogg` -> Ogg Vorbis

如果没有扩展名，或扩展名不受支持，函数会回退为 WAV，并输出一条告警日志（`"Unknown audio file extension, defaulting to WAV format"`）。

扩展名匹配不区分大小写，因此像 `.WAV` 这类大写扩展名也会被正确识别。

### 另请参阅

- `cosyvoice_save_wav`
- `cosyvoice_audio_load_from_file`

## cosyvoice_audio_free

### 语法

```c
COSYVOICE_API void cosyvoice_audio_free(float* data);
```

### 说明

释放音频 API 分配的缓冲区。

### 参数

- `data`：待释放音频缓冲区。

### 返回值

无返回值。
