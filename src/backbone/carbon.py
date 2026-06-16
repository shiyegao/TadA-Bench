from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class CarbonModel(nn.Module):
    model_names = [
        "HuggingFaceBio/Carbon-500M",
        "HuggingFaceBio/Carbon-3B",
        "HuggingFaceBio/Carbon-8B",
    ]

    DTYPES = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }

    def __init__(
        self,
        model_name: str,
        del_special_tokens: bool = True,
        device=None,
        dtype: str | torch.dtype = torch.bfloat16,
        revision: str | None = None,
        expected_num_tokens: int | None = None,
    ):
        assert model_name in self.model_names, (
            f"Model name {model_name} not found in {self.model_names}"
        )
        super().__init__()
        self.device = torch.device(
            device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.del_special_tokens = del_special_tokens
        self.expected_num_tokens = expected_num_tokens
        self.revision = revision
        self.dtype = self.DTYPES.get(dtype, dtype) if isinstance(dtype, str) else dtype

        load_kwargs = {"trust_remote_code": True}
        if self.revision:
            load_kwargs["revision"] = self.revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            **load_kwargs,
            "torch_dtype": self.dtype,
        }
        try:
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        except ValueError:
            causal_lm = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.model = causal_lm.model

        self.model.config.use_cache = False
        self.model = self.model.to(self.device)

    @staticmethod
    def wrap_dna(seq: str) -> str:
        return f"<dna>{seq}</dna>"

    def forward(self, seqs: List[str]):
        texts = [self.wrap_dna(str(seq)) for seq in seqs]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.device)
        outputs = self.model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        embeddings = outputs.last_hidden_state
        if self.del_special_tokens:
            embeddings = embeddings[:, 1:-1]

        if (
            self.expected_num_tokens is not None
            and embeddings.shape[1] != self.expected_num_tokens
        ):
            raise ValueError(
                "Carbon token count mismatch: "
                f"expected {self.expected_num_tokens}, got {embeddings.shape[1]}. "
                "Check DNA tag handling and del_special_tokens."
            )
        return embeddings
