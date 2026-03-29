# cosyvoice-frontend.h API Reference

This page documents all public symbols declared in `include/cosyvoice-frontend.h`.

## COSYVOICE_API

### Syntax

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

### Description

Controls symbol visibility for frontend API functions.

## cosyvoice_text_ptr

### Syntax

```c
typedef struct cosyvoice_text
{
    uint32_t    length;
    const char* text;
}* cosyvoice_text_ptr;
```

### Description

Pointer to a normalized text container returned by frontend utilities.

### Fields

- `length`: Byte length of `text`, excluding null terminator.
- `text`: Pointer to a null-terminated UTF-8 string.

## cosyvoice_text_create

### Syntax

```c
COSYVOICE_API cosyvoice_text_ptr cosyvoice_text_create();
```

### Description

Allocates a normalized-text object.

### Returns

New text object on success; `NULL` on failure.

## cosyvoice_text_free

### Syntax

```c
COSYVOICE_API void cosyvoice_text_free(cosyvoice_text_ptr text);
```

### Description

Frees a text object created by `cosyvoice_text_create`.

### Parameters

- `text`: Text object to release.

## cosyvoice_frontend_util_text_normalize

### Syntax

```c
COSYVOICE_API void cosyvoice_frontend_util_text_normalize(
    const char* text,
    uint32_t    text_len,
    const char* locale,
    cosyvoice_text_ptr normalized_text
);
```

### Description

Normalizes input text for prompt processing.

### Parameters

- `text`: Input UTF-8 text buffer.
- `text_len`: Input length in bytes.
- `locale`: Optional locale hint; pass `NULL` for auto-detection.
- `normalized_text`: Output text object receiving normalized result.

## cosyvoice_frontend_context_t

### Syntax

```c
typedef struct cosyvoice_frontend_context* cosyvoice_frontend_context_t;
```

### Description

Opaque handle to a frontend context.

## cosyvoice_frontend_load_from_files

### Syntax

```c
COSYVOICE_API cosyvoice_frontend_context_t cosyvoice_frontend_load_from_files(
    const char* speech_tokenizer,
    const char* campplus
);
```

### Description

Loads frontend models from files on disk.

### Parameters

- `speech_tokenizer`: Path to speech-tokenizer model.
- `campplus`: Path to campplus model.

### Returns

Frontend context handle on success; `NULL` on failure.

## OrtEnv

### Syntax

```c
struct OrtEnv;
```

### Description

Forward declaration of ONNX Runtime environment type used by frontend loading APIs.

## OrtSessionOptions

### Syntax

```c
struct OrtSessionOptions;
```

### Description

Forward declaration of ONNX Runtime session options type used by frontend loading APIs.

## cosyvoice_frontend_load

### Syntax

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

### Description

Loads frontend models from in-memory blobs.

### Parameters

- `speech_tokenizer_data`: Pointer to speech-tokenizer model bytes.
- `speech_tokenizer_size`: Byte size of `speech_tokenizer_data`.
- `campplus_data`: Pointer to campplus model bytes.
- `campplus_size`: Byte size of `campplus_data`.
- `env`: Optional ONNX Runtime environment.
- `session_options`: Optional ONNX Runtime session options.

### Returns

Frontend context handle on success; `NULL` on failure.

## cosyvoice_frontend_prompt_speech

### Syntax

```c
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech(
    cosyvoice_frontend_context_t ctx,
    float* speech,
    uint32_t speech_len,
    uint32_t sample_rate,
    const char* prompt_text
);
```

### Description

Extracts prompt-speech features from one waveform. Requires audio API support for resampling and waveform processing.

### Parameters

- `ctx`: Frontend context handle.
- `speech`: Input mono PCM samples.
- `speech_len`: Number of input samples.
- `sample_rate`: Input sample rate in Hz.
- `prompt_text`: Optional text paired with the prompt speech.

### Returns

Prompt-speech handle on success; `NULL` on failure.

## cosyvoice_frontend_prompt_speech_direct

### Syntax

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

### Description

Extracts prompt-speech features from pre-resampled 16 kHz and 24 kHz waveforms.

### Parameters

- `ctx`: Frontend context handle.
- `speech_16k`: Input mono PCM at 16 kHz.
- `speech_16k_len`: Number of samples in `speech_16k`.
- `speech_24k`: Input mono PCM at 24 kHz.
- `speech_24k_len`: Number of samples in `speech_24k`.
- `prompt_text`: Optional text paired with the prompt speech.

### Returns

Prompt-speech handle on success; `NULL` on failure.

## cosyvoice_frontend_free

### Syntax

```c
COSYVOICE_API void cosyvoice_frontend_free(cosyvoice_frontend_context_t ctx);
```

### Description

Releases a frontend context and associated model resources.

### Parameters

- `ctx`: Frontend context handle to destroy.
