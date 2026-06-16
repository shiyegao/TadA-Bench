[2026-06-11 10:42:25 CST]

# Carbon Experiment Report

Start time: [2026-06-10 16:14:30 CST]
Current report timestamp: [2026-06-11 10:42:25 CST]
Experiment completion elapsed wall time: 13 h 00 m 52 s
Primary zero-shot GPU host: local RTX 5080 workstation, 16,303 MiB
Primary supervised GPU platform: H200 notebook environment, NVIDIA H200 143,771 MiB
Code worktree: `codex/carbon-eval`
Branch: `codex/carbon-eval`

## Summary

Carbon-500M was integrated and formally evaluated on the official TadA-Bench
future-round DNA validation and test splits for three protocols:
zero-shot DNA base-pair likelihood, frozen-backbone MLP, and full fine-tuning
through the MLP head. Each formal protocol was run with three repeats/seeds on
the full splits. The H200 supervised summary passed strict completeness checks:
3 seeds x 3 learning rates x val/test for MLP and FT, all at epoch 20.

The selected frozen MLP hyperparameter is `head=3e-05`, chosen by mean
validation Spearman across seeds. The selected FT hyperparameter is
`backbone=3e-05, head=3e-05`, also chosen by mean validation Spearman. Test
metrics below are reported only after validation-based selection.

Clarification added at [2026-06-11 10:42:25 CST]: this follows the paper
appendix setting of running the three learning rates `{3e-5, 1e-4, 3e-4}` and
selecting the best learning rate by validation performance on round 28. For this
Carbon report, "validation performance" is operationalized as validation
Spearman, the primary ranking metric. Test metrics are not used for LR
selection.

PT remains unreported as a formal Carbon benchmark score because the current
public training stack does not implement a prompt/prefix/PEFT pathway. Carbon-3B
and Carbon-8B were not run formally in this H200 batch; their previous smoke and
resource findings remain feasibility notes only.

## H200 Continuation Update

Update timestamp: [2026-06-11 05:15:22 CST]

The Carbon-500M full frozen-backbone MLP and FT experiments were moved from the
RTX 5080 host to H200 notebooks after resource and network checks. The first
H100 and first H200 notebooks exposed notebook control-plane instability during
large payload upload, so the final transfer path used the Jupyter Contents API
with project-persistent storage.

Remote H200 setup facts:

| Item | Value |
| --- | --- |
| H200 image | `ngc-pytorch:25.02-cuda12.8.0-py3` |
| GPU observed | NVIDIA H200, 143,771 MiB |
| Notebook resources | 1xH200 jobs used 20 CPU / 200 GiB / 1 GPU; 8xH200 requests remained pending due CPU/memory scheduling |
| Project storage | project-persistent notebook storage |
| Offline HF cache | `hf_home`, about 2.3 GB after Carbon-500M, Qwen tokenizer, and TadA-Bench cache |
| Wheelhouse | `transformers==4.48.1`, `datasets==3.6.0`, `huggingface_hub==0.31.2`, `tokenizers==0.21.1`, `dill==0.3.8`, `multiprocess==0.70.16`, `xxhash==3.7.0` |
| Network status from notebook | Hugging Face/GitHub unreachable from the notebook; internal package mirror reachable; operator proxies unavailable from notebook |
| Cache transfer | 2,356,449,280-byte tar uploaded as 71 x 32 MiB chunks; SHA256 `0217523b54a63a7ae19d66d141232d9f260c25a76ac9413f003d15a8160d08c0` |

H200 smoke/profile evidence:

| Model | Protocol | Scope | Status | Evidence |
| --- | --- | --- | --- | --- |
| Carbon-500M | MLP smoke | 4 train / 2 val / 2 test | Completed on H200 | `results/metrics/smoke/TadABench_future_round_MLP_Carbon-500M_smoke_{val,test}.json` |
| Carbon-500M | FT probe | 2 train / 2 val / 2 test, `frozen_backbone=False` | Completed on H200 | `results/metrics/probe/TadABench_future_round_FT_Carbon-500M_probe_{val,test}.json` |
| Carbon-500M | MLP profile | 4096 train / 2048 val / 2048 test | Completed on H200 in about 14 s | val Spearman 0.0561254851; test Spearman 0.0738434856 |
| Carbon-500M | FT profile | 4096 train / 2048 val / 2048 test, batch128 | Completed on H200 | confirmed `Params of backbone: 254` and no OOM |

Formal H200 supervised run status at [2026-06-11 05:15:22 CST]:

| Protocol | Repeats/grid | Status | Evidence |
| --- | ---: | --- | --- |
| Frozen MLP | 3 seeds x 3 LRs x val/test | Completed, all epoch 20 | `results/carbon_summary_h200_full`, `complete=true` |
| Full fine-tuning | 3 seeds x 3 LRs x val/test | Completed, all epoch 20 | `results/carbon_summary_h200_full`, `complete=true` |

Important protocol detail: each generated full MLP seed config expands three
head learning rates, `3e-5`, `1e-4`, and `3e-4`, through `scripts/run.py`. FT was
run as nine single-LR jobs with matched backbone/head learning rates,
`3e-5`, `1e-4`, and `3e-4`, using `scripts/run_config_once.py`. Final reporting
selects hyperparameters by validation Spearman only. Test metrics generated at
epoch 0 are pre-training records and are not used for selection or formal
reporting.

Additional independent review was performed for this H200 continuation:

- Protocol audit: no train/val/test leakage or test-label training path found;
  smoke/profile/FT probe are explicit subsets; full MLP uses the official full
  DNA splits 729,302 / 148,014 / 149,884. Carbon tokenization is correctly
  documented as 84 DNA 6-mer tokens after removing `<dna>` tag tokens, not 501
  base-pair positions.
- Machine audit: H200 resources and project storage were sufficient for
  Carbon-500M frozen MLP and FT. H100 likely has enough compute/memory for
  Carbon-500M MLP, but the observed H100 failure was control-plane/upload
  related. PT, Carbon-3B, and Carbon-8B remain separate feasibility items.
- Final artifact audit initially found that H200 MLP/FT JSONs were present only
  on remote project storage, not in the local worktree. The H200 artifacts were
  then archived and synced locally. The local archive contains 45 files:
  9 summary files, 18 MLP metric JSONs, and 18 FT metric JSONs.

## Formal Result Table

The table below contains only full-split, three-repeat results suitable for the
benchmark page.

| Model | Protocol | Modality | Repeats | Split | N | Spearman | Recall@10% | NDCG@10% | Std. Dev. |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| HuggingFaceBio/Carbon-500M | Zero-shot base-pair log-likelihood, FNS `score_sequence()` | DNA | 3 | Validation | 148,014 | 0.0215409230 | 0.1034948050 | 0.2390359055 | effectively 0 across deterministic repeats |
| HuggingFaceBio/Carbon-500M | Zero-shot base-pair log-likelihood, FNS `score_sequence()` | DNA | 3 | Test | 149,884 | 0.0212969419 | 0.1052561455 | 0.3166686785 | 0 across deterministic repeats |
| HuggingFaceBio/Carbon-500M | Frozen-backbone MLP, selected LR `head=3e-05` | DNA | 3 | Validation | 148,014 | 0.0848162000 | 0.1161790000 | 0.2725570000 | sp 0.00217063; recall 0.00166222; ndcg 0.000155549 |
| HuggingFaceBio/Carbon-500M | Frozen-backbone MLP, selected LR `head=3e-05` | DNA | 3 | Test | 149,884 | 0.0843220000 | 0.1044570000 | 0.3265990000 | sp 0.000832107; recall 0.00180238; ndcg 0.00104235 |
| HuggingFaceBio/Carbon-500M | Full fine-tuning, selected LR `backbone=head=3e-05` | DNA | 3 | Validation | 148,014 | 0.0518556000 | 0.1063060000 | 0.2573190000 | sp 0.0143257; recall 0.00717642; ndcg 0.0061694 |
| HuggingFaceBio/Carbon-500M | Full fine-tuning, selected LR `backbone=head=3e-05` | DNA | 3 | Test | 149,884 | 0.0474591000 | 0.0930207000 | 0.3169440000 | sp 0.0142165; recall 0.00707979; ndcg 0.00894357 |

Formal artifacts:

| Artifact | Location |
| --- | --- |
| Zero-shot prediction CSVs | local GPU run: `predictions/future_round/TadABench_future_round_Carbon_likelihood_Carbon-500M_repeat{1,2,3}_{val,test}.csv` |
| Zero-shot metric JSONs | local GPU run: `results/metrics/future_round/TadABench_future_round_Carbon_likelihood_Carbon-500M_repeat{1,2,3}_{val,test}.json` |
| H200 MLP/FT metric JSONs | remote project storage: `results/metrics/future_round/TadABench_future_round_{MLP,FT}_Carbon-500M_seed*_{val,test}.json` |
| H200 summary CSV/MD | remote project storage: `results/carbon_summary_h200_full/` |
| H200 completeness status | `results/carbon_summary_h200_full/carbon_summary_status.json`, `complete=true` |
| Local H200 audit archive | `docs/artifacts/carbon_h200_artifacts_20260611_0511.tar.gz`, SHA256 `c98d04391fbe6b8c97104e86d739efb7814db6f4aa2065b52b306f56a30e3698` |
| Extracted H200 artifact paths | The audit archive expands to `results/carbon_summary_h200_full/` and `results/metrics/future_round/`; extracted copies are intentionally not tracked separately |

Completeness checks:

| Check | Result |
| --- | --- |
| Validation row count | 148,014 rows in every repeat CSV and JSON |
| Test row count | 149,884 rows in every repeat CSV and JSON |
| Repeat coverage | zero-shot repeats 1, 2, and 3; MLP/FT seeds 1, 2, and 3 for every LR in the three-LR grid |
| Subset guard | zero-shot formal CSV/JSONs carry `is_subset=false`; MLP/FT full-split status is verified by full configs plus `num_examples=148014/149884` in every final JSON |
| Audit columns | zero-shot prediction CSVs include `seed`, `repeat`, `revision`, `protocol`, `max_samples`, `is_subset`; MLP/FT selected rows include `seed`, `hyperparameter_id`, `config_path`, and `num_examples` |
| Summary strict mode | `scripts/summarize_carbon_results.py --strict` completed with `"complete": true` and no errors |
| MLP/FT final epoch guard | 18 MLP JSONs and 18 FT JSONs present; `non_epoch20=[]` |
| Local H200 artifact guard | local extracted archive has 36 MLP/FT final JSONs; seeds `[1,2,3]`; splits `[val,test]`; epoch set `[20]`; sample-count set `[148014,149884]` |
| LR-grid strict guard | updated strict check requires the full `{3e-5,1e-4,3e-4}` MLP/FT hyperparameter grid, val/test splits, and seeds `[1,2,3]`; rerun on local H200 artifacts returned `complete=true` |

## Learning-Rate Sweep and Selection

The formal MLP/FT rows above are not single-LR runs. They are selected from the
full three-LR sweep by mean validation Spearman across seeds.

| Protocol | Hyperparameter | Val Spearman mean | Test Spearman mean | Selected? | Reason |
| --- | --- | ---: | ---: | --- | --- |
| Frozen-backbone MLP | `head=3e-05` | 0.0848161967 | 0.0843219542 | Yes | highest validation Spearman |
| Frozen-backbone MLP | `head=0.0001` | 0.0797429800 | 0.0862806424 | No | lower validation Spearman |
| Frozen-backbone MLP | `head=0.0003` | 0.0775764999 | 0.0877024607 | No | lower validation Spearman, even though test Spearman is higher |
| Full fine-tuning | `backbone=head=3e-05` | 0.0518556383 | 0.0474590752 | Yes | highest validation Spearman |
| Full fine-tuning | `backbone=head=0.0001` | 0.0401412914 | 0.0278433788 | No | lower validation Spearman |
| Full fine-tuning | `backbone=head=0.0003` | 0.0000000000 | 0.0000000000 | No | lower validation Spearman |

## Full Experiment Matrix

This table includes every requested protocol and model-size tier we evaluated or
considered. Rows marked "not formal" must not be posted as benchmark scores.

| Model | Protocol | Requested 3 repeats? | Execution status | Evidence | Reason if not formal |
| --- | --- | ---: | --- | --- | --- |
| Carbon-500M | Zero-shot FNS likelihood | Yes | Completed formal full-split 3 repeats | Full val/test CSV + JSON, strict summary complete | Not applicable |
| Carbon-500M | Frozen-backbone MLP | Yes | Completed formal full-split 3 seeds x 3 LRs | `results/carbon_summary_h200_full`, selected `head=3e-05` by validation Spearman | Not applicable |
| Carbon-500M | Prompt tuning / PT | Yes | Not run | Code review and subagent audit | No prompt/prefix/PEFT/PT module or config exists in the public training stack |
| Carbon-500M | Full fine-tuning / FT | Yes | Completed formal full-split 3 seeds x 3 LRs | `results/carbon_summary_h200_full`, selected `backbone=head=3e-05` by validation Spearman | Not applicable |
| Carbon-3B | Zero-shot FNS likelihood | Yes | Not formal; 128-example probe completed | `TadABench_future_round_Carbon-3B_likelihood_probe.log` | Full three-repeat val/test run was not scheduled after resource checks; model loads, but full 3B scoring would add substantial extra GPU time beyond the 500M formal result |
| Carbon-3B | Frozen-backbone MLP | Yes | Not formal; smoke and profile completed | smoke retry passed; 512/256/256 profile at batch32 ran about 5.6 batch/s, ~180 seq/s | Estimated at roughly 28 h per learning-rate run and over 250 h for 3 seeds x 3 LRs x 20 epochs on this single GPU |
| Carbon-3B | PT | Yes | Not run | Code review and subagent audit | Same PT implementation gap as Carbon-500M |
| Carbon-3B | FT | Yes | Not run | Resource inference from 3B storage/profile and 500M FT OOM | 3B full FT is not defensible on a single 16.3 GiB RTX 5080 without PEFT/checkpointing/offload |
| Carbon-8B | Any protocol | Yes | Not run | Hugging Face metadata: safetensors total ~16.51 GB | Source bf16 weights alone exceed the practical single-card VRAM budget once runtime overhead/activations are included |

## Resource Findings

| Item | Value |
| --- | ---: |
| RTX 5080 GPU memory | 16,303 MiB |
| H200 GPU memory | 143,771 MiB |
| Free disk on RTX 5080 target after runs | about 1.5 TiB |
| H200 offline HF cache | about 2.3 GB |
| Formal prediction artifact size | 704 MB |
| Carbon-500M safetensors | 1.024 GB |
| Carbon-3B safetensors | 6.903 GB |
| Carbon-8B safetensors | 16.511 GB |
| Official DNA train/val/test rows | 729,302 / 148,014 / 149,884 |

Throughput and feasibility notes:

| Probe | Result |
| --- | --- |
| Carbon-500M zero-shot, batch256 | Full three-repeat val/test completed in about one hour wall time |
| Carbon-500M MLP profile, batch128 | 4096 train + 2048 val + 2048 test, 1 epoch, 33.59 s; about 8 batch/s |
| Carbon-500M MLP profile, batch256 | Same subset, 36.17 s; no throughput improvement |
| Carbon-500M FT probe, batch1 | Unfrozen backbone backward path works on 2 train examples |
| Carbon-500M FT profile on H200, batch128 | 4096 train + 2048 val + 2048 test, 1 epoch completed; backbone trainable |
| Carbon-500M full MLP on H200 | 3 seeds x 3 LRs completed in about 6.3 h wall time using three 1xH200 notebooks |
| Carbon-500M full FT on H200 | 3 seeds x 3 LRs completed in about 4.75 h wall time using nine 1xH200 notebooks |
| Carbon-3B smoke | Initial proxy download timed out on shard 3; retry with `HF_HUB_DOWNLOAD_TIMEOUT=120` succeeded |
| Carbon-3B MLP profile, batch32 | 512 train + 256 val + 256 test, 1 epoch, 30.73 s; about 5.6 batch/s |

## Protocol Notes

Zero-shot Carbon FNS scoring:

- DNA sequences are scored with Carbon FNS `score_sequence()`.
- The prediction score is `mean(log(actual_probs.clamp_min(1e-12)))`.
- Higher values mean higher model likelihood; no negative sign is applied.
- This matches the Carbon model-card examples for base-pair scoring.
- The formal zero-shot script does not use labels when generating predictions;
  labels are loaded only for metric calculation and internal audit CSVs.

Carbon hidden-state MLP:

- The backbone path wraps DNA as `<dna>{sequence}</dna>`, uses
  `add_special_tokens=False`, and removes the two tag tokens before MLP input.
- For 501 bp TadA sequences, Carbon yields 84 DNA 6-mer tokens after removing
  tags.
- 501 is not divisible by 6, so Carbon's tokenizer handles the trailing partial
  6-mer according to its own DNA tokenization behavior. This is valid for the
  current implementation but remains a protocol detail to document when comparing
  with other Carbon evaluations.

Hyperparameter selection:

- Frozen MLP configs preserve the three learning rates `{3e-5, 1e-4, 3e-4}` for
  the head.
- FT uses matched backbone/head learning rates `{3e-5, 1e-4, 3e-4}`. The
  backbone is unfrozen (`frozen_backbone=False`) and optimizer groups include
  both `backbone` and `head`.
- Generated configs exist for seeds 1, 2, and 3 for Carbon-500M and Carbon-3B.
- `scripts/summarize_carbon_results.py` selects MLP and FT hyperparameters by
  mean validation Spearman across seeds when full metrics are available.
- Test rows are not used for hyperparameter selection in the summarizer.

## Independent Subagent Audit Summary

Independent read-only subagents reviewed the implementation, machine setup, and
experiment protocol.

Zero-shot audit:

- No blocker found for full-split defaults, label leakage in scoring, DNA FNS
  usage, or score direction.
- Confirmed that default formal zero-shot paths use `all.DNA.val` and
  `all.DNA.test` unless `--max_samples` is explicitly provided.
- Confirmed that `score_sequence()` + mean log actual-base probability is the
  correct direction; higher scores rank higher.
- Flagged repeat/schema weaknesses and subset ambiguity. Fixes applied:
  repeat-aware default run IDs, `repeat/seed/revision/protocol/max_samples/is_subset`
  metadata in outputs, README updates, and strict summary checks.

MLP/PT/FT audit:

- Confirmed Carbon MLP configs use full splits, correct model dimensions, bf16
  backbone dtype, and frozen-backbone mode.
- Flagged that README originally showed single-seed Carbon MLP commands. Fix
  applied: README now points formal Carbon MLP usage to generated seed configs.
- Flagged missing validation-based hyperparameter selection. Fix applied:
  `scripts/summarize_carbon_results.py`.
- Identified PT as unimplemented in the current public training stack.
- Identified FT as more than a `frozen_backbone=False` switch; optimizer groups,
  memory strategy, and runtime strategy were required. H200 follow-up completed
  the full formal FT grid with matched backbone/head learning rates.

Final H200 artifact audit:

- Initial finding: the local worktree could independently verify only
  zero-shot; H200 MLP/FT formal artifacts were still remote-only. This was a
  real audit gap, not a model/protocol failure.
- Remediation: one completed 1xH200 notebook was restarted briefly, the remote
  project-storage artifacts were archived, downloaded, checksummed, extracted,
  and the notebook was stopped again.
- Synced evidence: 45 files in the local archive, including
  `results/carbon_summary_h200_full/*` plus 36 MLP/FT final metric JSONs.
- Machine check after sync: MLP files = 18, FT files = 18; seeds = 1/2/3;
  splits = val/test; epoch = 20 for every JSON; sample counts are 148,014 or
  149,884 with no bad rows.
- No finding of label leakage, subset execution, wrong score sign, negative
  learning rates, or Carbon tag/token misuse. PT, Carbon-3B, and Carbon-8B
  remain correctly excluded from formal benchmark scores.

## Task Timing

| Phase | Time / Duration |
| --- | --- |
| Start | [2026-06-10 16:14:30 CST] |
| Zero-shot and initial integration | completed before H200 supervised continuation |
| H200 supervised continuation and artifact sync | completed by [2026-06-11 05:15:22 CST] |
| Total elapsed wall time | 13 h 00 m 52 s |

## Source References

- Carbon model collection and model cards: <https://huggingface.co/collections/HuggingFaceBio/carbon>, <https://huggingface.co/HuggingFaceBio/Carbon-500M>, <https://huggingface.co/HuggingFaceBio/Carbon-3B>
- Carbon source repository and evaluation examples: <https://github.com/huggingface/carbon>
- TadA-Bench public site: <https://tada-bench.github.io/>
- TadA-Bench paper/reference setting used by subagent audit: <https://arxiv.org/html/2606.02624v1>
