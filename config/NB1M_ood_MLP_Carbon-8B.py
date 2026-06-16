from src.constants import OFFICIAL_HF_DATASET_REVISION

task = "run"


# Dataset
dataset_type = "RegressionDataset"
train_val_test = True
huggingface_dataset = "JinGao/TadA-Bench"
huggingface_revision = OFFICIAL_HF_DATASET_REVISION
del_special_tokens = True


# Embed from files
use_embed_mapper = False
mem_cache = False
frozen_backbone = True
dtype = {"head": "fp32", "backbone": "bf16"}
embed_name = "HuggingFaceBio/Carbon-8B"
model_revision = "31f9f2bf4d721a89c5bc11111822387a2d07f5ba"
seq_type = "DNA"
length = 501


# Evaluation and saving
eval_before_train = True
eval_interval = 1
save_interval = -1
save_dir = "ckpt/future_round"
prediction_dir = "predictions/future_round"
metric_dir = "results/metrics/future_round"
run_id = "TadABench_future_round_MLP_Carbon-8B"
eval_after_train = True


# Optimizer and learning rate
num_epochs = 20
batch_size = 16
optimizer_type = "AdamW"
learning_rate = [{"head": 3e-5}, {"head": 1e-4}, {"head": 3e-4}]
scheduler_kwargs = {
    "type": "CosineAnnealingLR",
    "is_epoch": True,
    "num_warmup_steps": 1,
    "num_training_steps": num_epochs,
}
weight_decay = {"head": 1e-4}
test_batch_size = 32
loss_type = ["mse"]
evaluation = [
    [
        "sp",
        "recall_at_10pct",
        "ndcg_at_10pct",
    ]
]


# Model
head_model_type = "MLP"
num_tokens = 84 + 2 * (0 if del_special_tokens else 1)
embed_dim = 4096
hidden_sizes = [
    [int(501 * 4096 * 16 / num_tokens / embed_dim)],
]
num_layers = 2
activation = "ReLU"


# Wandb config
use_wandb = False
tag = "MLP_Carbon-8B"
wandb_proj_name = None
entity = None
