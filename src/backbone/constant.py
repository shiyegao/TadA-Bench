models = [
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "InstaDeepAI/nucleotide-transformer-500m-1000g",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/agro-nucleotide-transformer-1b",
    "esm3_sm_open_v1",
    "esmc_300m",
    "esmc_600m",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t48_15B_UR50D",
    "facebook/esm2_t36_3B_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t6_8M_UR50D",
    "HuggingFaceBio/Carbon-500M",
    "HuggingFaceBio/Carbon-3B",
    "HuggingFaceBio/Carbon-8B",
]


def MODEL_MAP(model_name, seq_type, cfg, pretrained: bool = True):
    del_special_tokens = getattr(cfg, "del_special_tokens", True)
    device = getattr(cfg, "device", None)
    assert model_name in models, f"Model {model_name} not found in {models}."
    if model_name in [
        "HuggingFaceBio/Carbon-500M",
        "HuggingFaceBio/Carbon-3B",
        "HuggingFaceBio/Carbon-8B",
    ]:
        if seq_type != "DNA":
            raise ValueError(f"Carbon models require seq_type='DNA', got {seq_type}.")
        from .carbon import CarbonModel

        dtype_cfg = getattr(cfg, "dtype", {})
        backbone_dtype = dtype_cfg.get("backbone", "bf16") if isinstance(dtype_cfg, dict) else "bf16"
        return CarbonModel(
            model_name,
            del_special_tokens=del_special_tokens,
            device=device,
            dtype=backbone_dtype,
            revision=getattr(cfg, "model_revision", None),
            expected_num_tokens=getattr(cfg, "num_tokens", None),
        )
    elif model_name in ["esm3_sm_open_v1", "esmc_300m", "esmc_600m"]:
        from .esm3 import ESM3Model

        return ESM3Model(
            model_name, del_special_tokens=del_special_tokens, device=device
        )
    elif model_name in [
        "facebook/esm2_t33_650M_UR50D",
        "facebook/esm2_t48_15B_UR50D",
        "facebook/esm2_t36_3B_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t6_8M_UR50D",
    ]:
        from .esm2 import ESM2Model

        return ESM2Model(
            model_name,
            pretrained,
            del_special_tokens=del_special_tokens,
            device=device,
        )
    elif model_name in [
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "InstaDeepAI/nucleotide-transformer-500m-1000g",
        "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "InstaDeepAI/agro-nucleotide-transformer-1b",
    ]:
        from .nucleotidetransformer import NucleotideTransformerModel

        return NucleotideTransformerModel(
            model_name, del_special_tokens=del_special_tokens, device=device
        )
    else:
        raise ValueError(f"Model {model_name} not found in {models}.")
