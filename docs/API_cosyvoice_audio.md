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

- `cosyvoice_save_wav`
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
