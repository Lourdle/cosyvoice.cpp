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

## cosyvoice_audio_encoder_t

### 语法

```c
typedef struct cosyvoice_audio_encoder* cosyvoice_audio_encoder_t;
```

### 说明

音频编码器的不透明句柄。

## cosyvoice_audio_encoding_format_t

### 语法

```c
typedef enum cosyvoice_audio_encoding_format
{
    COSYVOICE_AUDIO_ENCODING_FORMAT_WAV = 0,
    COSYVOICE_AUDIO_ENCODING_FORMAT_MP3,
    COSYVOICE_AUDIO_ENCODING_FORMAT_AAC,
    COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC,
    COSYVOICE_AUDIO_ENCODING_FORMAT_M4A,
    COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS,
    COSYVOICE_AUDIO_ENCODING_FORMAT_COUNT
} cosyvoice_audio_encoding_format_t;
```

### 说明

内存音频编码器支持的音频格式。

FFmpeg 后端在公共 API 中暴露 `wav`、`mp3`、`aac`、`flac`、`m4a`、`opus`，但具体可用子集取决于当前链接的 FFmpeg 运行时。`m4a` 是本项目提供的便捷扩展，不属于 OpenAI Speech 标准。

## cosyvoice_audio_encoding_format_supported

### 语法

```c
COSYVOICE_API bool cosyvoice_audio_encoding_format_supported(cosyvoice_audio_encoding_format_t format);
```

### 说明

检查指定格式是否被音频编码器支持。

### 参数

- `format`：待检查的格式。

### 返回值

支持返回 `true`，否则返回 `false`。

## cosyvoice_audio_supported_encoding_formats

### 语法

```c
COSYVOICE_API const char* cosyvoice_audio_supported_encoding_formats(void);
```

### 说明

返回一个以逗号分隔的 NUL 终止字符串，列出运行时音频后端支持的编码格式。该值在运行时计算（例如在使用 FFmpeg 后端时会探测可用的编码器），主要用于 CLI/server 帮助文本或 UI 显示。

### 返回值

返回值指向库内部拥有的字符串，生命周期与进程相同。

## cosyvoice_audio_encoder_create

### 语法

```c
COSYVOICE_API cosyvoice_audio_encoder_t cosyvoice_audio_encoder_create(uint32_t sample_rate);
```

### 说明

创建一个内存音频编码器。

### 参数

- `sample_rate`：输入采样率（Hz）。

### 返回值

成功返回编码器句柄，否则返回 `NULL`。

## cosyvoice_audio_encoder_destroy

### 语法

```c
COSYVOICE_API void cosyvoice_audio_encoder_destroy(cosyvoice_audio_encoder_t encoder);
```

### 说明

销毁由 `cosyvoice_audio_encoder_create` 创建的编码器。

### 参数

- `encoder`：编码器句柄，允许为 `NULL`。

## cosyvoice_audio_encoder_encode

### 语法

```c
COSYVOICE_API bool cosyvoice_audio_encoder_encode(
    cosyvoice_audio_encoder_t encoder,
    const float* input,
    uint32_t length,
    cosyvoice_audio_encoding_format_t format
);
```

### 说明

将单声道浮点 PCM 编码为指定音频格式。

### 参数

- `encoder`：编码器句柄。
- `input`：输入单声道浮点 PCM 缓冲区。
- `length`：输入采样点数量。
- `format`：目标音频格式。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_audio_encoder_get_encoded_data

### 语法

```c
COSYVOICE_API void cosyvoice_audio_encoder_get_encoded_data(cosyvoice_audio_encoder_t encoder,
    const uint8_t** data,
    uint32_t* length
);
```

### 说明

获取上一次成功编码得到的输出数据。

### 参数

- `encoder`：编码器句柄。
- `data`：接收编码字节指针，归编码器所有。
- `length`：接收编码后字节长度。

### 备注

返回的 `data` 指针在下一次编码调用之前，或者编码器销毁之前保持有效。

## cosyvoice_audio_decoder_t

### 语法

```c
typedef struct cosyvoice_audio_decoder* cosyvoice_audio_decoder_t;
```

### 说明

内存音频解码器的不透明句柄。将编码音频从内存缓冲区解码为单声道 float PCM，无需访问文件系统。

## cosyvoice_audio_decoding_format_supported

### 语法

```c
COSYVOICE_API bool cosyvoice_audio_decoding_format_supported(cosyvoice_audio_encoding_format_t format);
```

### 说明

检查指定音频格式是否支持解码。

### 参数

- `format`：待测试的格式。

### 返回值

支持解码返回 `true`；否则返回 `false`。

## cosyvoice_audio_decoder_create

### 语法

```c
COSYVOICE_API cosyvoice_audio_decoder_t cosyvoice_audio_decoder_create(void);
```

### 说明

创建内存音频解码器。解码器会自动从缓冲区内容中检测音频格式。

### 返回值

成功返回解码器句柄；失败返回 `NULL`。

## cosyvoice_audio_decoder_destroy

### 语法

```c
COSYVOICE_API void cosyvoice_audio_decoder_destroy(cosyvoice_audio_decoder_t decoder);
```

### 说明

销毁由 `cosyvoice_audio_decoder_create` 创建的解码器。

### 参数

- `decoder`：解码器句柄。允许传入 `NULL`。

## cosyvoice_audio_decoder_decode

### 语法

```c
COSYVOICE_API bool cosyvoice_audio_decoder_decode(
    cosyvoice_audio_decoder_t decoder,
    const void*               input,
    uint32_t                  input_length
);
```

### 说明

将内存缓冲区中的编码音频解码为单声道 float PCM。解码器会自动从缓冲区内容中检测音频格式。解码成功后，调用 `cosyvoice_audio_decoder_get_decoded_data` 获取 PCM 数据。

### 参数

- `decoder`：解码器句柄。
- `input`：编码音频数据指针。
- `input_length`：编码数据长度（字节）。

### 返回值

成功返回 `true`；失败返回 `false`。

## cosyvoice_audio_decoder_get_decoded_data

### 语法

```c
COSYVOICE_API void cosyvoice_audio_decoder_get_decoded_data(
    cosyvoice_audio_decoder_t decoder,
    float**                   data,
    uint32_t*                 length,
    uint32_t*                 sample_rate
);
```

### 说明

获取上一次成功解码得到的 PCM 数据。返回的 `data` 指针在下一次解码调用之前，或者解码器销毁之前保持有效。调用者不得释放该数据。

### 参数

- `decoder`：解码器句柄。
- `data`：接收单声道 float PCM 缓冲区的指针（`[-1, 1]`）。
- `length`：接收采样点数。
- `sample_rate`：接收采样率（Hz）。

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

如果没有扩展名，或扩展名不受支持，函数会回退为 WAV，并输出一条告警日志（`"Unknown audio file extension, defaulting to WAV format"`）。

扩展名匹配不区分大小写，因此像 `.WAV` 这类大写扩展名也会被正确识别。

### 另请参阅

- `cosyvoice_audio_encoder_encode`
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
