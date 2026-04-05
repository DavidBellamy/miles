"""
DeepSeek V3 5-layer NVFP4 rollout + BF16 training script.

Usage:
    # EP=4 (default)
    python scripts/run_deepseek_v3_5layer_nvfp4.py

    # EP=2 DP=2
    TRAIN_EP=2 python scripts/run_deepseek_v3_5layer_nvfp4.py

    # Skip data/model preparation (if already done)
    python scripts/run_deepseek_v3_5layer_nvfp4.py --skip-prepare

    # Custom model/data dirs
    python scripts/run_deepseek_v3_5layer_nvfp4.py --model-dir /shared/models --data-dir /root/datasets
"""

import argparse
import os
from pathlib import Path

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "DeepSeek-V3-5layer"
MODEL_ORG = "chwan"
MODEL_TYPE = "deepseek-v3-5layer"
NUM_GPUS = 4


def prepare(args):
    model_dir = args.model_dir
    data_dir = args.data_dir
    train_ep = args.train_ep

    U.exec_command(f"mkdir -p {model_dir} {data_dir}")
    U.exec_command(f"hf download {MODEL_ORG}/{MODEL_NAME} --local-dir {model_dir}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=data_dir)

    U.fp8_cast_bf16(
        path_src=f"{model_dir}/{MODEL_NAME}",
        path_dst=f"{model_dir}/{MODEL_NAME}-bf16/",
    )

    nvfp4_dir = Path(model_dir) / f"{MODEL_NAME}-NVFP4"
    if not (nvfp4_dir / "model.safetensors.index.json").exists():
        U.exec_command(
            f"python tools/convert_hf_to_nvfp4.py "
            f"--model-dir {model_dir}/{MODEL_NAME}-bf16 "
            f"--save-dir {nvfp4_dir}"
        )

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        hf_checkpoint=f"{model_dir}/{MODEL_NAME}-bf16",
        multinode=False,
        extra_args=(
            f"--tensor-model-parallel-size 1 "
            f"--pipeline-model-parallel-size 1 "
            f"--expert-model-parallel-size {train_ep} "
            "--expert-tensor-parallel-size 1 "
        ),
        dir_dst=model_dir,
    )

    nvfp4_link = Path(model_dir) / f"{MODEL_NAME}-NVFP4-current"
    U.exec_command(f"ln -sfn {nvfp4_dir} {nvfp4_link}")


def execute(args):
    model_dir = args.model_dir
    data_dir = args.data_dir
    train_ep = args.train_ep

    ckpt_args = (
        f"--hf-checkpoint {model_dir}/{MODEL_NAME}-NVFP4-current "
        f"--ref-load {model_dir}/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        f"--prompt-data {data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 1 "
        "--num-steps-per-rollout 4 "
        "--balance-data "
        "--rollout-max-response-len 100 "
    )

    eval_args = (
        f"--eval-prompt-data aime {data_dir}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 8 "
        "--eval-max-response-len 32768 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {train_ep} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-precision-aware-optimizer "
        "--exp-avg-dtype bf16 "
        "--exp-avg-sq-dtype bf16 "
        "--main-grads-dtype bf16 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.55 "
        "--sglang-tp-size 4 "
        "--sglang-ep-size 4 "
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 1 "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-server-concurrency 1024 "
        "--sglang-max-running-requests 256 "
        "--sglang-chunked-prefill-size 1024 "
        "--sglang-cuda-graph-max-bs 256 "
        "--sglang-quantization modelopt_fp4 "
        "--sglang-attention-backend trtllm_mla "
        "--sglang-moe-runner-backend flashinfer_trtllm "
        "--sglang-kv-cache-dtype fp8_e4m3 "
        """--sglang-model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' """
        "--sglang-enable-nan-detection "
        "--sglang-enable-fp32-lm-head "
    )

    misc_args = (
        "--bf16 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--grad-reduce-in-bf16 "
        "--attention-softmax-in-fp32 "
        "--update-weight-buffer-size 536870912 "
        f"--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--use-fault-tolerance "
        "--disable-weights-backuper "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3 5-layer NVFP4 training")
    parser.add_argument("--model-dir", default="/shared/zhichen/models")
    parser.add_argument("--data-dir", default="/root/datasets")
    parser.add_argument("--train-ep", type=int, default=int(os.environ.get("TRAIN_EP", "4")))
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    if not args.skip_prepare:
        prepare(args)

    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    execute(args)


if __name__ == "__main__":
    main()
