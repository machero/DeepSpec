# Local launch mirrors the repo's node launcher, not standard torchrun
# semantics. eval.py spawns one worker per visible GPU by itself.
# init_dist defaults to single-node local run when MASTER_ADDR/MASTER_PORT and
# RANK/WORLD_SIZE are unset. Total GPU workers come from CUDA_VISIBLE_DEVICES.

# Match this to the target model used by the draft checkpoint.
# Supported target models include:
#   Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B
#   Qwen/Qwen3.5-27B
#   google/gemma-4-12B-it
target_name_or_path=Qwen/Qwen3-4B

# Training writes checkpoints under ~/checkpoints/<project_name>/<exp_name>/step_*.
# Use step_latest for the most recent checkpoint, or replace it with step_<N>.
# For Qwen3.5-27B, use the corresponding exp_name, e.g.:
#   draft_name_or_path=${HOME}/checkpoints/deepspec/dspark_block7_qwen3_5_27b/step_latest
draft_name_or_path=${HOME}/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path ${target_name_or_path} \
    --draft_name_or_path ${draft_name_or_path}
