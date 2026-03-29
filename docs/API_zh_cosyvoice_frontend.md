# cosyvoice-frontend.h API 参考

本文档覆盖 `include/cosyvoice-frontend.h` 中声明的全部公开符号。

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

控制前端 API 的导入导出可见性。

## cosyvoice_text_ptr

### 语法

```c
typedef struct cosyvoice_text
{
    uint32_t    length;
    const char* text;
}* cosyvoice_text_ptr;
```

### 说明

文本归一化结果对象指针。

### 成员

- `length`：`text` 的字节长度（不含结尾空字符）。
- `text`：UTF-8 字符串指针。

## cosyvoice_text_create

### 语法

```c
COSYVOICE_API cosyvoice_text_ptr cosyvoice_text_create();
```

### 说明

创建归一化文本对象。

### 参数

无。

### 返回值

成功返回对象指针，失败返回 `NULL`。

## cosyvoice_text_free

### 语法

```c
COSYVOICE_API void cosyvoice_text_free(cosyvoice_text_ptr text);
```

### 说明

释放文本对象。

### 参数

- `text`：待释放对象。

### 返回值

无返回值。

## cosyvoice_frontend_util_text_normalize

### 语法

```c
COSYVOICE_API void cosyvoice_frontend_util_text_normalize(
    const char* text,
    uint32_t    text_len,
    const char* locale,
    cosyvoice_text_ptr normalized_text
);
```

### 说明

对输入文本进行归一化处理。

### 参数

- `text`：输入 UTF-8 文本。
- `text_len`：输入字节长度。
- `locale`：地区提示，传 `NULL` 表示自动检测。
- `normalized_text`：输出文本对象。

### 返回值

无返回值。

## cosyvoice_frontend_context_t

### 语法

```c
typedef struct cosyvoice_frontend_context* cosyvoice_frontend_context_t;
```

### 说明

前端上下文不透明句柄。

## cosyvoice_frontend_load_from_files

### 语法

```c
COSYVOICE_API cosyvoice_frontend_context_t cosyvoice_frontend_load_from_files(
    const char* speech_tokenizer,
    const char* campplus
);
```

### 说明

从文件路径加载前端模型并创建上下文。

### 参数

- `speech_tokenizer`：speech tokenizer 模型路径。
- `campplus`：campplus 模型路径。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

## OrtEnv

### 语法

```c
struct OrtEnv;
```

### 说明

ONNX Runtime 环境类型前置声明。

## OrtSessionOptions

### 语法

```c
struct OrtSessionOptions;
```

### 说明

ONNX Runtime 会话配置类型前置声明。

## cosyvoice_frontend_load

### 语法

```c
COSYVOICE_API cosyvoice_frontend_context_t cosyvoice_frontend_load(
    const void* speech_tokenizer_data,
    size_t speech_tokenizer_size,
    const void* campplus_data,
    size_t campplus_size,
    const struct OrtEnv* env,
    const struct OrtSessionOptions* session_options
);
```

### 说明

从内存数据块加载前端模型。

### 参数

- `speech_tokenizer_data`：speech tokenizer 模型内存指针。
- `speech_tokenizer_size`：speech tokenizer 数据大小（字节）。
- `campplus_data`：campplus 模型内存指针。
- `campplus_size`：campplus 数据大小（字节）。
- `env`：可选 ORT 环境对象。
- `session_options`：可选 ORT 会话配置对象。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

## cosyvoice_frontend_prompt_speech

### 语法

```c
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech(
    cosyvoice_frontend_context_t ctx,
    float* speech,
    uint32_t speech_len,
    uint32_t sample_rate,
    const char* prompt_text
);
```

### 说明

从单路波形提取提示语音特征。

### 参数

- `ctx`：前端上下文。
- `speech`：输入单声道 PCM。
- `speech_len`：输入采样点数量。
- `sample_rate`：输入采样率（Hz）。
- `prompt_text`：可选提示文本。

### 返回值

成功返回提示语音句柄，失败返回 `NULL`。

### 备注

该接口依赖音频 API 支持。

## cosyvoice_frontend_prompt_speech_direct

### 语法

```c
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech_direct(
    cosyvoice_frontend_context_t ctx,
    float* speech_16k,
    uint32_t speech_16k_len,
    float* speech_24k,
    uint32_t speech_24k_len,
    const char* prompt_text
);
```

### 说明

直接使用 16k/24k 波形提取提示语音特征。

### 参数

- `ctx`：前端上下文。
- `speech_16k`：16kHz 输入波形。
- `speech_16k_len`：16kHz 波形采样点数量。
- `speech_24k`：24kHz 输入波形。
- `speech_24k_len`：24kHz 波形采样点数量。
- `prompt_text`：可选提示文本。

### 返回值

成功返回提示语音句柄，失败返回 `NULL`。

## cosyvoice_frontend_free

### 语法

```c
COSYVOICE_API void cosyvoice_frontend_free(cosyvoice_frontend_context_t ctx);
```

### 说明

释放前端上下文及其资源。

### 参数

- `ctx`：待释放前端上下文。

### 返回值

无返回值。
