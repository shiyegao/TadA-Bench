from src.constants import OFFICIAL_HF_DATASET_REVISION

task = "run"


# Dataset
dataset_type = "RegressionDataset"
train_val_test = True
huggingface_dataset = "JinGao/TadABench-1M"
huggingface_revision = OFFICIAL_HF_DATASET_REVISION
del_special_tokens = True
max_train_samples = 8
max_val_samples = 4
max_test_samples = 4
num_workers = 0


# Embed from files
use_embed_mapper = False
mem_cache = False
frozen_backbone = True
dtype = {"head": "fp32", "backbone": "bf16"}
embed_name = "facebook/esm2_t12_35M_UR50D"
seq_type = "AA"
length = 167


# Evaluation and saving
eval_before_train = True
eval_interval = 1
save_interval = -1
save_dir = "ckpt/smoke"


# Optimizer and learning rate
num_epochs = 1
batch_size = 2
optimizer_type = "AdamW"
learning_rate = {"head": 1e-4}
scheduler_kwargs = {
    "type": "CosineAnnealingLR",
    "is_epoch": True,
    "num_warmup_steps": 1,
    "num_training_steps": num_epochs,
}
weight_decay = {"head": 1e-4}
test_batch_size = 2
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
num_tokens = length + 2 * (0 if del_special_tokens else 1)
embed_dim = 480
hidden_sizes = [[64]]
num_layers = 2
activation = "ReLU"


# Wandb config
use_wandb = False
tag = "MLP_ESM2-35M_smoke"
wandb_proj_name = None
entity = None


seed = 1
prediction_dir = "predictions/smoke"
metric_dir = "results/metrics/smoke"
run_id = "NB1M_ood_MLP_ESM2-35M_smoke"
eval_after_train = True
