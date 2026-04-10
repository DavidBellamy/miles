"""Agent V2 launcher (GLM-4.7 Full / 355B-A32B): Miles <-> Harbor agent orchestration.

Same model architecture as GLM-4.5-355B-A32B. Targets 16 x 8-GPU H200 nodes (sci-h200).

Usage:
    python run-glm47-full.py --num-nodes 16
    python run-glm47-full.py --num-nodes 16 --skip-prepare
    python run-glm47-full.py --num-nodes 16 --mode debug_rollout_only
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "glm4.5-355B-A32B"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths
    skip_prepare: bool = False
    model_name: str = "GLM-4.7"
    hf_checkpoint: str = "/models/zai-org/GLM-4.7"
    ref_load: str = "/models/zai-org/GLM-4.7_torch_dist"
    save_dir: str = "/root/GLM-4.7-Full_agent_v2/"
    max_seq_len: int = 16384
    prompt_data: str = "/root/swe_train.jsonl"

    # Agent settings
    agent_server_url: str = os.environ.get(
        "AGENT_SERVER_URL", os.environ.get("SWE_AGENT_URL", "http://agent_env:11000")
    )
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", "/root/harbor_tasks")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", socket.gethostname())
    miles_host_ip: str = os.environ.get(
        "MILES_HOST_IP", socket.gethostbyname(socket.gethostname())
    )

    # W&B settings
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "glm47-full-agentic")
    wandb_team: str = os.environ.get("WANDB_TEAM", "")
    wandb_run_name: str = "glm47-full-swe-tito"

    # Prometheus settings
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "glm47-full-swe-tito"


def cleanup():
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def prepare(args: ScriptArgs):
    """Convert HF checkpoint to torch_dist format (multinode for 355B).

    The conversion tool requires world_size <= num_layers (92 for this model).
    Cap conversion nodes so total GPUs don't exceed 92.
    """
    max_convert_nodes = 92 // args.num_gpus_per_node  # 11 for 8 GPUs/node
    convert_nodes = min(args.num_nodes, max_convert_nodes)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=True,
        num_nodes=convert_nodes,
        dir_dst=str(Path(args.ref_load).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        "--save-interval 100 "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-temperature 0.8 "
        "--rollout-max-response-len 8192 "
        f"--max-seq-len {args.max_seq_len} "
        "--global-batch-size 64 "
        "--balance-data "
    )

    # Training parallelism: TP=4, PP=2, EP chosen as largest divisor of 160 that fits.
    # 92 layers / PP=2 = 46 layers per stage.
    # 160 experts → EP must divide 160. Divisors ≤DP: {1,2,4,5,8,10,16,20,...}
    tp, pp = 4, 2
    total_gpus = args.num_nodes * args.num_gpus_per_node
    dp = total_gpus // (tp * pp)
    assert total_gpus % (tp * pp) == 0, (
        f"total GPUs ({total_gpus}) must be divisible by TP*PP ({tp * pp})"
    )
    num_experts = 160
    # Pick largest divisor of num_experts that is <= dp
    ep = max(d for d in range(1, dp + 1) if num_experts % d == 0)

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {pp} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ep} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
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
    )

    # SGLang: TP=8 (1 node/engine, 16 engines). No cross-node TP.
    # 355B/8*2B → 82.53 GB/GPU. mem-fraction-static is total budget (model+KV).
    # Model needs 82.53/141 = 58.5%, so fraction must be >0.59.
    # 0.80 → budget=112.8 GB, KV=30.3 GB, CUDA overhead=28.2 GB.
    # CUDA graphs disabled: avoids OOM during capture (only 28 GB headroom).
    # TP=16 cross-node fails: ranks on remote nodes die during init, breaking
    # Gloo/NCCL connections. TP=8 avoids this by keeping all ranks on one node.
    sglang_world_size = min(8, total_gpus)
    # Round down to nearest multiple of gpus_per_node that divides total_gpus
    sglang_world_size = (sglang_world_size // args.num_gpus_per_node) * args.num_gpus_per_node
    while sglang_world_size > 0 and total_gpus % sglang_world_size != 0:
        sglang_world_size -= args.num_gpus_per_node
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.80 "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * 2048} "
        "--sglang-tool-call-parser glm47 "
        "--sglang-reasoning-parser glm45 "
        "--sglang-disable-cuda-graph "
        "--use-miles-router "
        "--sglang-router-port 31000 "
        "--session-server-port 30000 "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path swe_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm47 "
        "--chat-template-path autofix "
        "--use-session-server "
        "--session-server-port 30000 "
        "--tito-allowed-append-roles user tool "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {total_gpus} "
        "--use-fault-tolerance "
    )

    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""

    wandb_args = ""
    if args.wandb_key:
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {args.wandb_project} "
            f"--wandb-group {args.wandb_run_name} "
            f"--wandb-key {args.wandb_key} "
        )
        if args.wandb_team:
            wandb_args += f"--wandb-team {args.wandb_team} "

    prometheus_args = ""
    if args.use_prometheus:
        prometheus_args = (
            "--use-prometheus "
            f"--prometheus-port {args.prometheus_port} "
            f"--prometheus-run-name {args.prometheus_run_name} "
        )

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )

    miles_root = U.repo_base_dir

    extra_env_vars = {
        "PYTHONPATH": f"{args.megatron_path}:{SCRIPT_DIR}:{miles_root}",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_SERVER_URL": args.agent_server_url,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        "MILES_HOST_IP": args.miles_host_ip,
        # Disable NVLS — 355B model leaves too little memory for NVLink SHARP multicast buffers
        "NCCL_NVLS_ENABLE": "0",
        # Work around SGLang deprecation mapping bug: SGL_DISABLE_...=true maps
        # to SGLANG_ENABLE_...=true (same value, wrong semantics). Setting to
        # "false" makes the mapping produce SGLANG_ENABLE_...=false → check disabled.
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    }

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
