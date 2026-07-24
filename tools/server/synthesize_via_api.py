#!/usr/bin/env python3

import argparse
import pathlib
import sys


def normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if not url:
        raise ValueError("base_url is empty")

    if url.endswith("/v1"):
        return url
    return url + "/v1"


def guess_output_path(response_format: str) -> str:
    ext = response_format.lower()
    if ext == "pcm":
        return "tts_output.pcm"
    return f"tts_output.{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call cosyvoice-server via OpenAI API and save/play speech audio."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Server base URL, with or without /v1 (default: http://127.0.0.1:8080)",
    )
    parser.add_argument("--api-key", default="", help="Bearer API key (optional)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--voice", default="alloy", help="Voice name")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--instructions", default="", help="Optional instructions")
    parser.add_argument(
        "--response-format",
        default=None,
        help="Requested audio format (if omitted, inferred from --output extension)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (> 0)")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed extension (uint32)")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature extension (> 0)")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top_k extension (>= 0)")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top_p extension ([0, 1])")
    parser.add_argument("--win-size", type=int, default=None, help="Optional win_size extension (> 0)")
    parser.add_argument("--tau-r", type=float, default=None, help="Optional tau_r extension (>= 0)")
    parser.add_argument(
        "--min-token-text-ratio",
        type=float,
        default=None,
        help="Optional min_token_text_ratio extension (>= 0)",
    )
    parser.add_argument(
        "--max-token-text-ratio",
        type=float,
        default=None,
        help="Optional max_token_text_ratio extension (>= 0)",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming TTS (play while receiving if format is wav)")
    parser.add_argument("--chunk-tokens", type=int, default=None, help="Tokens per streaming chunk")
    parser.add_argument("--no-play", action="store_true", help="Disable real-time playback even with --stream")
    parser.add_argument(
        "--output",
        default="",
        help="Output audio file path (default: auto by response_format)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--requests",
        type=int,
        default=1,
        help="Number of sequential requests to send (default: 1)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int:
    if args.speed <= 0:
        print("Error: --speed must be > 0", file=sys.stderr)
        return 2
    if args.requests <= 0:
        print("Error: --requests must be > 0", file=sys.stderr)
        return 2

    if args.seed is not None and (args.seed < 0 or args.seed > 0xFFFFFFFF):
        print("Error: --seed must be in [0, 4294967295]", file=sys.stderr)
        return 2
    if args.temperature is not None and args.temperature <= 0:
        print("Error: --temperature must be > 0", file=sys.stderr)
        return 2
    if args.top_k is not None and args.top_k < 0:
        print("Error: --top-k must be >= 0", file=sys.stderr)
        return 2
    if args.top_p is not None and (args.top_p < 0 or args.top_p > 1):
        print("Error: --top-p must be in [0, 1]", file=sys.stderr)
        return 2
    if args.win_size is not None and args.win_size <= 0:
        print("Error: --win-size must be > 0", file=sys.stderr)
        return 2
    if args.tau_r is not None and args.tau_r < 0:
        print("Error: --tau-r must be >= 0", file=sys.stderr)
        return 2
    if args.min_token_text_ratio is not None and args.min_token_text_ratio < 0:
        print("Error: --min-token-text-ratio must be >= 0", file=sys.stderr)
        return 2
    if args.max_token_text_ratio is not None and args.max_token_text_ratio < 0:
        print("Error: --max-token-text-ratio must be >= 0", file=sys.stderr)
        return 2
    if args.chunk_tokens is not None and args.chunk_tokens < 0:
        print("Error: --chunk-tokens must be >= 0", file=sys.stderr)
        return 2
    if (
        args.min_token_text_ratio is not None
        and args.max_token_text_ratio is not None
        and args.max_token_text_ratio < args.min_token_text_ratio
    ):
        print("Error: --max-token-text-ratio must be >= --min-token-text-ratio", file=sys.stderr)
        return 2

    return 0


def request_and_stream(base_url, api_key, body, response_format, output_path, play, timeout):
    """Send streaming TTS request, play chunks in real-time, and save to file."""
    import requests

    url = base_url + "/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=body, stream=True, timeout=timeout)
    resp.raise_for_status()

    _play_stream = None
    _pyaudio_instance = None
    _all_data = bytearray()
    _wav_header_buf = bytearray()

    try:
        _use_pyaudio = play and response_format == "wav"
        if _use_pyaudio:
            try:
                import pyaudio as _pa
            except ImportError:
                print("Warning: pyaudio not installed. Install with: pip install pyaudio", file=sys.stderr)
                _use_pyaudio = False
            else:
                _pyaudio_instance = _pa.PyAudio()

        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            _all_data.extend(chunk)

            if _use_pyaudio:
                if _play_stream is None:
                    _wav_header_buf.extend(chunk)
                    if len(_wav_header_buf) >= 44:
                        import struct
                        channels = struct.unpack_from('<H', bytes(_wav_header_buf), 22)[0]
                        sample_rate = struct.unpack_from('<I', bytes(_wav_header_buf), 24)[0]
                        bps = struct.unpack_from('<H', bytes(_wav_header_buf), 34)[0]
                        sw = bps // 8
                        fmt = _pyaudio_instance.get_format_from_width(sw)
                        _play_stream = _pyaudio_instance.open(
                            format=fmt, channels=channels, rate=sample_rate, output=True
                        )
                        if len(_wav_header_buf) > 44:
                            _play_stream.write(bytes(_wav_header_buf[44:]))
                else:
                    _play_stream.write(chunk)

        output_path.write_bytes(bytes(_all_data))
        print(f"Saved {len(_all_data)} bytes to {output_path}")
    finally:
        if _play_stream:
            _play_stream.stop_stream()
            _play_stream.close()
        if _pyaudio_instance:
            _pyaudio_instance.terminate()


def request_and_save(client, kwargs, response_format, output_path):
    """Non-streaming request via OpenAI SDK."""
    audio = client.audio.speech.create(**kwargs)
    audio_bytes = audio.read()
    output_path.write_bytes(audio_bytes)
    return len(audio_bytes)


def main() -> int:
    args = parse_args()
    code = validate_args(args)
    if code != 0:
        return code

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package is not installed. Run: pip install openai", file=sys.stderr)
        return 2

    try:
        base_url = normalize_base_url(args.base_url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    api_key = args.api_key if args.api_key else "sk-no-auth-required"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)

    request_kwargs = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "speed": args.speed,
    }
    if args.instructions:
        request_kwargs["instructions"] = args.instructions

    extension_body = {}
    if args.seed is not None:
        extension_body["seed"] = args.seed
    if args.temperature is not None:
        extension_body["temperature"] = args.temperature
    if args.top_k is not None:
        extension_body["top_k"] = args.top_k
    if args.top_p is not None:
        extension_body["top_p"] = args.top_p
    if args.win_size is not None:
        extension_body["win_size"] = args.win_size
    if args.tau_r is not None:
        extension_body["tau_r"] = args.tau_r
    if args.min_token_text_ratio is not None:
        extension_body["min_token_text_ratio"] = args.min_token_text_ratio
    if args.max_token_text_ratio is not None:
        extension_body["max_token_text_ratio"] = args.max_token_text_ratio
    if args.stream:
        extension_body["stream"] = True
    if args.chunk_tokens is not None:
        extension_body["chunk_tokens"] = args.chunk_tokens
    if extension_body:
        request_kwargs["extra_body"] = extension_body

    response_format = args.response_format
    if response_format is None:
        if args.output:
            suffix = pathlib.Path(args.output).suffix
            if suffix:
                response_format = suffix.lstrip(".").lower()
            else:
                response_format = "wav"
        else:
            response_format = "wav"

    request_kwargs["response_format"] = response_format

    output = args.output if args.output else guess_output_path(response_format)
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    play = args.stream and not args.no_play

    total_bytes = 0
    saved_files = []
    for i in range(args.requests):
        if args.requests == 1:
            curr_path = output_path
        else:
            curr_path = output_path.with_name(
                f"{output_path.stem}_{i + 1:03d}{output_path.suffix}"
            )

        try:
            if args.stream:
                # Build the full request body (kwargs + extra_body)
                body = {k: v for k, v in request_kwargs.items() if k != "extra_body"}
                if "extra_body" in request_kwargs:
                    body.update(request_kwargs["extra_body"])
                request_and_stream(
                    base_url=base_url,
                    api_key=api_key,
                    body=body,
                    response_format=response_format,
                    output_path=curr_path,
                    play=play,
                    timeout=args.timeout,
                )
            else:
                total = request_and_save(client, request_kwargs, response_format, curr_path)
                total_bytes += total
                saved_files.append(curr_path)
                print(f"Saved {total} bytes to {curr_path}")
        except Exception as exc:
            print(f"Request failed (#{i + 1}): {exc}", file=sys.stderr)
            return 1

    if args.requests > 1:
        print(f"Saved {len(saved_files)} files, total {total_bytes} bytes")

    print(f"OpenAI base_url: {base_url}")
    if extension_body:
        print("Applied extension fields:", ", ".join(sorted(extension_body.keys())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
