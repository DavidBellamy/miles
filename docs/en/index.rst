Miles Documentation
====================

miles is an LLM post-training framework designed by `Radixark AI <https://radixark.ai>`_ for scaling RL training, with deep optimizations for large-scale MoE models, agentic rollout, fault tolerance, and production-grade training workflows.

Core capabilities:

- **High-Performance Training**: Efficient training in various modes by deeply integrating Megatron-LM (training backend) with SGLang (inference backend), supporting both colocate and disaggregated deployment.
- **Flexible Data Generation**: Arbitrary training data generation workflows through custom rollout functions, server-based engines, and agentic multi-turn tool-calling support.
- **Large-Scale MoE Support**: Production-tested on models up to 744B parameters with expert parallelism, DeepEP, and FP8 rollout.
- **Advanced RL Features**: GRPO/GSPO/PPO/REINFORCE++, dynamic sampling, partial rollout, TIS, R3 rollout routing replay, speculative decoding, P2P weight transfer, and more.

Supported Models
-----------------

miles provides out-of-the-box support for a wide range of open-source models:

- **GLM series**: GLM-5 (744B-A40B), GLM-4.7 (Flash), GLM-4.5 (355B-A32B, 106B-A12B), GLM-4 (9B, 32B);
- **Qwen3.5 series**: 4B, 9B, 27B, 35B-A3B;
- **Qwen3 series**: Qwen3-Next-80B-A3B, Qwen3-235B-A22B, Qwen3-30B-A3B, Qwen3 dense (0.6B/1.7B/4B/8B/14B/32B), Qwen3-4B-Instruct-2507;
- **Qwen2.5 series**: 0.5B, 1.5B, 3B, 7B, 32B;
- **DeepSeek series**: DeepSeek V3, V3.1, DeepSeek R1;
- **Kimi K2**: Instruct, Thinking;
- **Moonlight-16B-A3B**, **MiMo-7B-RL**;
- **gpt-oss-20B**;
- **Llama 3**: 3.1-8B-Instruct, 3.2-3B-Instruct.

You can find model configs under ``scripts/models/`` and launcher scripts under ``scripts/`` (``run_*.py`` recommended, ``run-*.sh`` for legacy workflows). Adding a new model architecture is straightforward — see the :doc:`advanced/arch-support-beyond-megatron` guide.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/quick_start.md
   get_started/usage.md
   get_started/customization.md
   get_started/gen_endpoint.md
   get_started/oai_endpoint.md
   get_started/qa.md

.. toctree::
   :maxdepth: 1
   :caption: Agentic

   agentic/chat_template_verification.md

.. toctree::
   :maxdepth: 1
   :caption: Dense

   examples/qwen3-4B.md
   examples/glm4-9B.md

.. toctree::
   :maxdepth: 1
   :caption: MoE

   examples/qwen3-30B-A3B.md
   examples/glm4.5-355B-A32B.md
   examples/deepseek-r1.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced/miles-router.md
   advanced/speculative-decoding.md
   advanced/fault-tolerance.md
   advanced/arch-support-beyond-megatron.md
   advanced/pd-disaggregation.md
   advanced/p2p-weight-transfer.md
   advanced/miles_server_args.md

.. toctree::
   :maxdepth: 1
   :caption: Other Usage

   examples/qwen3-4b-base-openhermes.md

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/debug.md
   developer_guide/migration.md

.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platform_support/amd_tutorial.md

.. toctree::
   :maxdepth: 1
   :caption: Blogs

   blogs/release_v0.1.0.md
   blogs/introducing_miles.md
