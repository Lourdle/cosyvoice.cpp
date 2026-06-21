import re
import yaml
import pathlib
import argparse

import torch
import gguf

from transformers import AutoTokenizer
from transformers import AutoConfig


def unknown_tag_constructor(loader, tag_suffix, node):
    # 直接返回该 YAML 节点的数据，不做处理
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    else:
        return None


yaml.SafeLoader.add_multi_constructor('!', unknown_tag_constructor)

def get_vocab(tokenizer_path: str):
    # Load tokenizer and add special tokens metadata
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]", "<|endofsystem|>",
            "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
            "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
            "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
            "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
            "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
            "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
            "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
            "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
            "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
            "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
            "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
            "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
            "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
            "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
            "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
            "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
            "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
            "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
            "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
            "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    vocab_size = len(tokenizer.vocab)

    pre_tok = tokenizer._tokenizer.pre_tokenizer
    split_repr = repr(pre_tok) 
    match = re.search(r'pattern=Regex\("(.+?)"\)', split_repr)
    if match is None:
        raise ValueError(f"Failed to extract tokenizer pre-tokenizer regex from: {split_repr}")
    tokpre_regex = match.group(1)

    reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
    added_vocab = tokenizer.get_added_vocab()

    added_tokens_decoder = tokenizer.added_tokens_decoder
    tokens = []
    toktypes = []

    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
        else:
            token: str = reverse_vocab[i]
            if token in added_vocab and added_tokens_decoder[i].special:
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
            tokens.append(token)

    return tokens, toktypes, tokpre_regex


def _is_broadcastable_to(shape_small, shape_big) -> bool:
    # Standard numpy/torch broadcasting, aligning on the right.
    if len(shape_small) > len(shape_big):
        return False
    pad = (1,) * (len(shape_big) - len(shape_small))
    shape_small = pad + tuple(shape_small)
    for a, b in zip(shape_small, shape_big):
        if a != 1 and a != b:
            return False
    return True


def _reshape_g_for_broadcast(g: torch.Tensor, v: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim % v.dim()

    if g.shape == v.shape:
        return g

    if g.dim() == 1 and g.numel() == v.size(dim):
        shape = [1] * v.dim()
        shape[dim] = v.size(dim)
        return g.reshape(shape)

    if _is_broadcastable_to(g.shape, v.shape):
        if g.dim() < v.dim():
            return g.reshape((1,) * (v.dim() - g.dim()) + tuple(g.shape))
        return g

    raise ValueError(
        f"g cannot be broadcast to v: g={tuple(g.shape)} v={tuple(v.shape)} dim={dim}"
    )


def _norm_except_dim(v: torch.Tensor, dim: int, eps: float = 1e-12) -> torch.Tensor:
    dim = dim % v.dim()
    reduce_dims = tuple(d for d in range(v.dim()) if d != dim)
    if len(reduce_dims) == 0:
        n = v.abs()
    else:
        n = torch.linalg.vector_norm(v, ord=2, dim=reduce_dims, keepdim=True)
    return n.clamp_min(eps)


def _g_score(g: torch.Tensor, v: torch.Tensor, dim: int) -> int:
    """Higher score = more likely to be g for weight_norm."""
    try:
        gg = _reshape_g_for_broadcast(g, v, dim)
    except Exception:
        return -1

    dim = dim % v.dim()
    score = 0
    # g should be singleton on all non-dim axes; reward that.
    for d in range(v.dim()):
        if d == dim:
            continue
        if gg.shape[d] == 1:
            score += 2
        else:
            score -= 3

    # g should match v on dim axis (after reshape), or be 1 (broadcasted scalar-ish)
    if gg.shape[dim] == v.shape[dim]:
        score += 2
    elif gg.shape[dim] == 1:
        score += 0
    else:
        score -= 5

    # Prefer lower-rank/parameter-count g.
    score -= gg.numel() // 1024
    return score


def fold_parametrizations_weight_norm_state_dict(state_dict: dict, dim: int = 0) -> dict:
    """Fold parametrizations-based weight_norm (original0/original1) into plain .weight."""
    suffix0 = ".parametrizations.weight.original0"
    suffix1 = ".parametrizations.weight.original1"

    out = {}
    consumed = set()

    for k in list(state_dict.keys()):
        if k in consumed:
            continue

        if k.endswith(suffix0):
            base = k[:-len(suffix0)]
            k0 = base + suffix0
            k1 = base + suffix1
            if k1 not in state_dict:
                raise KeyError(f"Missing key for parametrizations pair: {k1}")

            a = state_dict[k0]
            b = state_dict[k1]
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"Expected tensors for {k0}/{k1}")

            # Decide which is g and which is v.
            sa = _g_score(a, b, dim)
            sb = _g_score(b, a, dim)
            if sa > sb:
                g, v = a, b
            elif sb > sa:
                g, v = b, a
            else:
                # Common convention fallback: original0=g original1=v
                g, v = a, b

            g = _reshape_g_for_broadcast(g, v, dim)
            w = v * (g / _norm_except_dim(v, dim))
            out[base + ".weight"] = w
            consumed.add(k0)
            consumed.add(k1)
            continue

        if k.endswith(suffix1):
            continue

        out[k] = state_dict[k]

    return out


def convert_cosyvoice_to_gguf(yaml_config_path: str, llm_model_path: str, blank_llm_path: str, flow_model_path: str, hift_model_path: str, gguf_model_path: str, tag: str, ftype: tuple[torch.dtype, gguf.GGMLQuantizationType, gguf.LlamaFileType]):
    # Load YAML configuration
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load LLM model checkpoint
    llm_state_dict = torch.load(llm_model_path, map_location='cpu')
    llm_state_dict = {k.replace('llm.model.model.', ''): v for k, v in llm_state_dict.items()}
    llm_state_dict.pop('llm.model.lm_head.weight', None)

    # Load Flow model checkpoint
    flow_state_dict = torch.load(flow_model_path, map_location='cpu')

    # Load HiFT model checkpoint
    hift_state_dict = torch.load(hift_model_path, map_location='cpu')
    hift_state_dict = fold_parametrizations_weight_norm_state_dict(hift_state_dict, dim=0)

    state_dict = {**llm_state_dict, **flow_state_dict, **hift_state_dict}
    gguf_model = gguf.GGUFWriter(gguf_model_path, "cosyvoice3" + (f"-{tag}" if tag else ""))
    gguf_model.add_file_type(ftype[2])

    for key, value in state_dict.items():
        raw_dtype = None
        if value.ndim == 2:
            if ftype[1] not in [None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]:
                try:
                    value = gguf.quantize(value.float().numpy(), ftype[1])
                    raw_dtype = ftype[1]
                except gguf.QuantError:
                    value = value.half().numpy()
            else:
                value = value.to(ftype[0]).numpy()
        else:
            value = value.float().numpy()
        gguf_model.add_tensor(name=key, tensor=value, raw_dtype=raw_dtype)

    # Load LLM config
    llm_config = AutoConfig.from_pretrained(blank_llm_path)
    speech_token_size = config["llm"]["speech_token_size"]
    sos = speech_token_size + 0
    task_id = speech_token_size + 2
    stop_token_ids = [speech_token_size + i for i in range(200)]

    # Add config
    gguf_model.add_string("cosyvoice.instruction_prefix", "You are a helpful assistant.")
    gguf_model.add_uint32("sample_rate", config["sample_rate"])
    gguf_model.add_int32("num_attention_heads", llm_config.num_attention_heads)
    gguf_model.add_int32("num_key_value_heads", llm_config.num_key_value_heads)
    gguf_model.add_int32("num_hidden_layers", llm_config.num_hidden_layers)
    gguf_model.add_float32("rms_norm_eps", llm_config.rms_norm_eps)
    gguf_model.add_float32("rope_theta", llm_config.rope_theta)
    gguf_model.add_int32("sos_token_id", sos)
    gguf_model.add_int32("task_token_id", task_id)
    gguf_model.add_array("stop_token_ids", stop_token_ids)
    # FSQ silent and breath tokens (CosyVoice3 codebook indices)
    gguf_model.add_array("silent_token_ids", [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323])
    gguf_model.add_key_value("mix_ratio", config["llm"]["mix_ratio"], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
    gguf_model.add_float32("sampling.top_p", config["llm"]["sampling"]["top_p"])
    gguf_model.add_int32("sampling.top_k", config["llm"]["sampling"]["top_k"])
    gguf_model.add_int32("sampling.win_size", config["llm"]["sampling"]["win_size"])
    gguf_model.add_float32("sampling.tau_r", config["llm"]["sampling"]["tau_r"])
    gguf_model.add_int32("token_mel_ratio", config["token_mel_ratio"])
    gguf_model.add_int32("pre_lookahead_layer.pre_lookahead_len", config["flow"]["pre_lookahead_layer"]["pre_lookahead_len"])
    gguf_model.add_float32("decoder.inference_cfg_rate", config["flow"]["decoder"]["cfm_params"]["content"]["inference_cfg_rate"])
    gguf_model.add_int32("decoder.estimator.heads", config["flow"]["decoder"]["estimator"]["heads"])
    gguf_model.add_int32("decoder.estimator.depth", config["flow"]["decoder"]["estimator"]["depth"])
    gguf_model.add_int32("decoder.estimator.mel_dim", config["flow"]["decoder"]["estimator"]["mel_dim"])
    gguf_model.add_int32("nb_harmonics", config["hift"]["nb_harmonics"])
    gguf_model.add_float32("nsf_alpha", config["hift"]["nsf_alpha"])
    gguf_model.add_float32("nsf_sigma", config["hift"]["nsf_sigma"])
    gguf_model.add_float32("lrelu_slope", config["hift"]["lrelu_slope"])
    gguf_model.add_float32("audio_limit", config["hift"]["audio_limit"])
    gguf_model.add_int32("nsf_voiced_threshold", config["hift"]["nsf_voiced_threshold"])
    gguf_model.add_int32("istft_params.n_fft", config["hift"]["istft_params"]["n_fft"])
    gguf_model.add_int32("istft_params.hop_len", config["hift"]["istft_params"]["hop_len"])
    gguf_model.add_int32("num_kernels", len(config["hift"]["resblock_kernel_sizes"]))
    gguf_model.add_array("upsample_rates", config["hift"]["upsample_rates"])
    n_resblock_kernels = len(config["hift"]["resblock_kernel_sizes"])
    for i, d in enumerate(config["hift"]["source_resblock_dilation_sizes"]):
        gguf_model.add_array(f"source_resblocks.{i}.dilations", d)
        for j, d in enumerate(config["hift"]["resblock_dilation_sizes"]):
            gguf_model.add_array(f"resblocks.{i * n_resblock_kernels + j}.dilations", d)

    # Add vocab
    tokens, toktypes, tokpre_regex = get_vocab(blank_llm_path)
    special_vocab = gguf.SpecialVocab(blank_llm_path, load_merges=True)
    gguf_model.add_string("tokenizer.model.type", "BPE")
    gguf_model.add_array("tokenizer.model.merges", special_vocab.merges)
    gguf_model.add_array("tokenizer.vocab.tokens", tokens)
    gguf_model.add_array("tokenizer.vocab.token_types", toktypes)
    gguf_model.add_string("tokenizer.pre_tokenizer.regex", tokpre_regex)

    # Save GGUF model
    gguf_model.write_header_to_file()
    gguf_model.write_kv_data_to_file()
    gguf_model.write_tensors_to_file()


def main():
    parser = argparse.ArgumentParser(description="Convert CosyVoice model to GGUF format.")
    parser.add_argument("--yaml_config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--llm_model", type=str, help="Path to LLM module checkpoint. The default is 'llm.pt' in the directory of the YAML config file.")
    parser.add_argument("--blank_llm", type=str, help="Path to blank LLM model. The default is 'CosyVoice-BlankEN' in the directory of the YAML config file.")
    parser.add_argument("--flow_model", type=str, help="Path to Flow module checkpoint. The default is 'flow.pt' in the directory of the YAML config file.")
    parser.add_argument("--hift_model", type=str, help="Path to HiFT module checkpoint. The default is 'hift.pt' in the directory of the YAML config file.")
    parser.add_argument("--gguf_model", type=str, required=True, help="Path to output GGUF model file.")
    parser.add_argument("--tag", type=str, help="Tag to identify the model variant in the GGUF file name.")
    parser.add_argument("--ftype", type=str, default="default", choices=["default", "f32", "f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1"], help="Data type for GGUF model tensors.")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.yaml_config).parent
    llm_model_path = args.llm_model if args.llm_model else model_dir / "llm.pt"
    blank_llm_path = args.blank_llm if args.blank_llm else model_dir / "CosyVoice-BlankEN"
    flow_model_path = args.flow_model if args.flow_model else model_dir / "flow.pt"
    hift_model_path = args.hift_model if args.hift_model else model_dir / "hift.pt"

    ftype_map = {
        "default": (None, None, gguf.LlamaFileType.GUESSED),
        "f32": (torch.float32, gguf.GGMLQuantizationType.F32, gguf.LlamaFileType.ALL_F32),
        "f16": (torch.float16, gguf.GGMLQuantizationType.F16, gguf.LlamaFileType.MOSTLY_F16),
        "q8_0": (torch.float32, gguf.GGMLQuantizationType.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0),
        "q5_0": (torch.float32, gguf.GGMLQuantizationType.Q5_0, gguf.LlamaFileType.MOSTLY_Q5_0),
        "q5_1": (torch.float32, gguf.GGMLQuantizationType.Q5_1, gguf.LlamaFileType.MOSTLY_Q5_1),
        "q4_0": (torch.float32, gguf.GGMLQuantizationType.Q4_0, gguf.LlamaFileType.MOSTLY_Q4_0),
        "q4_1": (torch.float32, gguf.GGMLQuantizationType.Q4_1, gguf.LlamaFileType.MOSTLY_Q4_1)
    }

    convert_cosyvoice_to_gguf(
        str(args.yaml_config),
        str(llm_model_path),
        str(blank_llm_path),
        str(flow_model_path),
        str(hift_model_path),
        str(args.gguf_model),
        args.tag,
        ftype_map[args.ftype],
    )


if __name__ == "__main__":
    main()
