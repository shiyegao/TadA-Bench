from src.constants import OFFICIAL_HF_DATASET_REVISION

task = "run"


# Dataset
dataset_type = "RegressionDataset"
train_val_test = True
huggingface_dataset = "JinGao/TadA-Bench"
huggingface_revision = OFFICIAL_HF_DATASET_REVISION
del_special_tokens = True
max_train_samples = 4
max_val_samples = 2
max_test_samples = 2
num_workers = 0


# Embed from files
use_embed_mapper = False
mem_cache = False
frozen_backbone = True
dtype = {"head": "fp32", "backbone": "bf16"}
embed_name = "HuggingFaceBio/Carbon-3B"
model_revision = "e4cc9617d29f1140ae3334c1a74dfb6ca6903d78"
seq_type = "DNA"
length = 501


# Evaluation and saving
eval_before_train = True
eval_interval = 1
save_interval = -1
save_dir = "ckpt/smoke"


# Optimizer and learning rate
num_epochs = 1
batch_size = 1
optimizer_type = "AdamW"
learning_rate = {"head": 1e-4}
scheduler_kwargs = {
    "type": "CosineAnnealingLR",
    "is_epoch": True,
    "num_warmup_steps": 1,
    "num_training_steps": num_epochs,
}
weight_decay = {"head": 1e-4}
test_batch_size = 1
loss_type = "mse"
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
embed_dim = 3072
hidden_sizes = [[64]]
num_layers = 2
activation = "ReLU"


# Wandb config
use_wandb = False
tag = "MLP_Carbon-3B_smoke"
wandb_proj_name = None
entity = None


seed = 1
prediction_dir = "predictions/smoke"
metric_dir = "results/metrics/smoke"
run_id = "NB1M_ood_MLP_Carbon-3B_smoke"
eval_after_train = True
