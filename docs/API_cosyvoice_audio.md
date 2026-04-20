# cosyvoice-audio.h API Reference

This page documents all public symbols declared in `include/cosyvoice-audio.h`.

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

Controls import/export attributes for audio helper APIs.

## cosyvoice_audio_encoder_t

### Syntax

```c
typedef struct cosyvoice_audio_encoder* cosyvoice_audio_encoder_t;
```

### Description

Opaque handle for the in-memory audio encoder.

## cosyvoice_audio_encoding_format_t

### Syntax

```c
typedef enum cosyvoice_audio_encoding_format
{
    COSYVOICE_AUDIO_ENCODING_FORMAT_WAV = 0,
    COSYVOICE_AUDIO_ENCODING_FORMAT_COUNT
} cosyvoice_audio_encoding_format_t;
```

### Description

Audio payload formats supported by the in-memory encoder.

## cosyvoice_audio_encoding_format_supported

### Syntax

```c
COSYVOICE_API bool cosyvoice_audio_encoding_format_supported(cosyvoice_audio_encoding_format_t format);
```

### Description

Checks whether a format is supported by the audio encoder.

### Parameters

- `format`: Format to test.

### Returns

`true` if supported; otherwise `false`.

## cosyvoice_audio_encoder_create

### Syntax

```c
COSYVOICE_API cosyvoice_audio_encoder_t cosyvoice_audio_encoder_create(uint32_t sample_rate);
```

### Description

Creates an in-memory audio encoder.

### Parameters

- `sample_rate`: Input sample rate in Hz.

### Returns

Encoder handle on success; otherwise `NULL`.

## cosyvoice_audio_encoder_destroy

### Syntax

```c
COSYVOICE_API void cosyvoice_audio_encoder_destroy(cosyvoice_audio_encoder_t encoder);
```

### Description

Destroys an encoder created by `cosyvoice_audio_encoder_create`.

### Parameters

- `encoder`: Encoder handle. `NULL` is allowed.

## cosyvoice_audio_encoder_encode

### Syntax

```c
COSYVOICE_API bool cosyvoice_audio_encoder_encode(
    cosyvoice_audio_encoder_t encoder,
    const float* input,
    uint32_t length,
	cosyvoice_audio_encoding_format_t format
);
```

### Description

Encodes mono float PCM data into a selected audio format.

### Parameters

- `encoder`: Encoder handle.
- `input`: Input mono float PCM buffer.
- `length`: Number of input samples.
- `format`: Target audio format.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_audio_encoder_get_encoded_data

### Syntax

```c
COSYVOICE_API void cosyvoice_audio_encoder_get_encoded_data(cosyvoice_audio_encoder_t encoder,
    const uint8_t** data,
    uint32_t* length
);
```

### Description

Gets the encoded payload from the last successful encode call.

### Parameters

- `encoder`: Encoder handle.
- `data`: Receives a pointer to encoded bytes owned by the encoder.
- `length`: Receives the encoded byte length.

### Remarks

The returned `data` pointer remains valid until the next encode call on the same encoder, or until the encoder is destroyed.

## cosyvoice_audio_load_from_file

### Syntax

```c
COSYVOICE_API bool cosyvoice_audio_load_from_file(
    const char* filename,
    float**     data,
    uint32_t*   length,
    uint32_t*   sample_rate
);
```

### Description

Loads an audio file and decodes it into mono float PCM data.

### Parameters

- `filename`: Input audio file path.
- `data`: Receives allocated sample buffer.
- `length`: Receives sample count.
- `sample_rate`: Receives detected sample rate in Hz.

### Returns

`true` on success; otherwise `false`.

### Remarks

Release `data` with `cosyvoice_audio_free`.

## cosyvoice_audio_resample

### Syntax

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

### Description

Resamples mono float PCM data to a target sample rate.

### Parameters

- `input`: Input sample buffer.
- `input_length`: Number of input samples.
- `input_sample_rate`: Input sample rate in Hz.
- `output`: Receives allocated output buffer.
- `output_length`: Receives number of output samples.
- `output_sample_rate`: Desired output sample rate in Hz.

### Returns

`true` on success; otherwise `false`.

### Remarks

Release `output` with `cosyvoice_audio_free`.

## cosyvoice_audio_save_to_file

### Syntax

```c
COSYVOICE_API bool cosyvoice_audio_save_to_file(
    const char*  filename,
    const float* data,
    uint32_t     length,
    uint32_t     sample_rate
);
```

### Description

Encodes mono float PCM data and writes it to an audio file.

### Parameters

- `filename`: Output file path.
- `data`: Input sample buffer.
- `length`: Number of samples.
- `sample_rate`: Output sample rate in Hz.

### Returns

`true` on success; otherwise `false`.

### Remarks

The encoder format is inferred from the output file extension:

- `.wav` -> WAV
- `.flac` -> FLAC
- `.mp3` -> MP3
- `.ogg` -> Ogg Vorbis

If the extension is missing or unsupported, the function falls back to WAV and emits a warning log message (`"Unknown audio file extension, defaulting to WAV format"`).

Extension matching is case-insensitive, so uppercase extensions (for example `.WAV`) are accepted.

### See also

- `cosyvoice_audio_encoder_encode`
- `cosyvoice_audio_load_from_file`

## cosyvoice_audio_free

### Syntax

```c
COSYVOICE_API void cosyvoice_audio_free(float* data);
```

### Description

Frees audio buffer memory allocated by audio APIs.

### Parameters

- `data`: Buffer pointer returned by `cosyvoice_audio_load_from_file` or `cosyvoice_audio_resample`.
