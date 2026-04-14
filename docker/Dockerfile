# 1. radixark/miles:dev (x86 CUDA 12)
#    use default settings
#
# 2. radixark/miles:dev-cu13-arm64 (arm64 CUDA 13 / GB300)
#    build-arg:ENABLE_CUDA_13=1 \
#    build-arg:SGLANG_IMAGE_TAG=v0.5.10-cu130 \
#    build-arg:WHEELS_TAG=cu130-aarch64 \
#
# 3. radixark/miles:dev-cu13-x86 (x86 CUDA 13 / B300, sm_103a)
#    build-arg:ENABLE_CUDA_13=1 \
#    build-arg:SGLANG_IMAGE_TAG=v0.5.10-cu130 \
#    build-arg:WHEELS_TAG=cu130-x86_64 \

ARG SGLANG_IMAGE_TAG=v0.5.10
FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG SGLANG_BRANCH=sglang-miles
ARG SGLANG_COMMIT=""

ARG MEGATRON_REPO=radixark/Megatron-LM
ARG MEGATRON_BRANCH=miles-main

ARG ENABLE_CUDA_13=0

ARG WHEELS_REPO=yueming-yuan/miles-wheels
ARG WHEELS_TAG=cu129-x86_64
ARG GITHUB_TOKEN=""

# When BUILD_FROM_SOURCE=1, skip pre-built wheels and compile flash-attn/apex from source.
# Use this when no pre-built wheels exist for the target platform (e.g. B300 x86 cu130).
ARG BUILD_FROM_SOURCE=0

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
# ethtool for network diagnostics
RUN apt install -y nvtop rsync dnsutils ethtool

# nccl-tests for diagnostics
RUN git clone https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests && \
    cd /tmp/nccl-tests && \
    git checkout ae98985f5599617be94042f4aa3637d10014ce89 && \
    make -j$(nproc) CUDA_HOME=/usr/local/cuda && \
    cp /tmp/nccl-tests/build/*_perf /usr/local/bin/ && \
    rm -rf /tmp/nccl-tests

# ====================================== Download pre-built wheels ============================================

RUN if [ "${BUILD_FROM_SOURCE}" = "0" ]; then \
      mkdir -p /tmp/wheels && \
      if [ -n "${GITHUB_TOKEN}" ]; then \
        API_RESP=$(curl -fSL -H "Authorization: token ${GITHUB_TOKEN}" "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}"); \
      else \
        API_RESP=$(curl -fSL "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}"); \
      fi && \
      echo "${API_RESP}" | python3 -c "import sys, json, subprocess; \
[subprocess.run(['curl', '-fSL', '-o', '/tmp/wheels/' + a['name'], a['browser_download_url']], check=True) \
 for a in json.load(sys.stdin)['assets'] if a['name'].endswith('.whl')]" && \
      ls -lh /tmp/wheels/; \
    else \
      mkdir -p /tmp/wheels; \
    fi

# ====================================== Python dependencies ============================================

# flash-attn (FA2): from pre-built wheel, or build from source
RUN if [ "${BUILD_FROM_SOURCE}" = "1" ]; then \
      MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation; \
    else \
      pip install /tmp/wheels/flash_attn-*.whl; \
    fi

# flash-attn hopper (FA3): Hopper-only (sm_90a). Coexists with FA2.
# From pre-built wheel, or build from source at the same commit as FA2.
RUN python_path=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p $python_path/flash_attn_3 && \
    if [ "${BUILD_FROM_SOURCE}" = "1" ]; then \
      git clone https://github.com/Dao-AILab/flash-attention.git --recursive /tmp/flash-attn-src && \
      cd /tmp/flash-attn-src && \
      git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && \
      MAX_JOBS=64 pip install hopper/ --no-build-isolation && \
      cp hopper/flash_attn_interface.py $python_path/flash_attn_3/ && \
      rm -rf /tmp/flash-attn-src; \
    else \
      pip install /tmp/wheels/flash_attn_3-*.whl && \
      curl -fSL https://raw.githubusercontent.com/Dao-AILab/flash-attention/fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89/hopper/flash_attn_interface.py \
        -o $python_path/flash_attn_3/flash_attn_interface.py; \
    fi

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.2
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx==25.6.0; \
    fi && \
    if ls /tmp/wheels/transformer_engine-*.whl 2>/dev/null | grep -q .; then \
      pip install /tmp/wheels/transformer_engine-*.whl; \
    elif [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip -v install --no-build-isolation "transformer_engine[core_cu13,pytorch]==2.12.0"; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"; \
    fi

# apex: from pre-built wheel, or build from source
RUN if [ "${BUILD_FROM_SOURCE}" = "1" ]; then \
      NVCC_APPEND_FLAGS="--threads 4" \
      pip -v install --disable-pip-version-check --no-cache-dir \
      --no-build-isolation \
      --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
      git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4; \
    else \
      pip install /tmp/wheels/apex-*.whl; \
    fi

RUN git clone https://github.com/${MEGATRON_REPO}.git --recursive -b ${MEGATRON_BRANCH} Megatron-LM && \
    cd Megatron-LM && \
    pip install -e .

RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 --no-cache-dir --force-reinstall
# RUN pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
RUN pip install git+https://github.com/yushengsu-thu/Megatron-Bridge.git@merged-megatron-0.16.0rc0-miles --no-deps --no-build-isolation
RUN pip install megatron-energon --no-deps
RUN pip install multi-storage-client --no-deps

COPY requirements.txt /tmp/requirements.txt
RUN rm -rf /usr/lib/python3/dist-packages/jwt /usr/lib/python3/dist-packages/PyJWT* && pip install -r /tmp/requirements.txt

# https://github.com/pytorch/pytorch/issues/168167
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
    pip install nvidia-cudnn-cu13==9.16.0.29; \
  else \
    pip install nvidia-cudnn-cu12==9.16.0.29; \
  fi

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Install sglang-miles ============================================

# Install sglang from sglang-miles branch
RUN cd /sgl-workspace/sglang && \
    git fetch origin ${SGLANG_BRANCH} && \
    if [ -n "${SGLANG_COMMIT}" ]; then \
      git checkout ${SGLANG_COMMIT}; \
    else \
      git checkout FETCH_HEAD; \
    fi && \
    pip install -e "python[all]" --no-deps

# ====================================== Install main package ============================================

ARG MILES_REPO=radixark/miles
ARG MILES_COMMIT=main
RUN git clone https://github.com/${MILES_REPO}.git /root/miles && \
    cd /root/miles && \
    git checkout ${MILES_COMMIT} && \
    pip install -e . --no-deps

# int4_qat: from pre-built wheel if available, otherwise build from in-tree source.
# setup.py generates sm_80/sm_90 binaries + compute_90 PTX for JIT on Blackwell.
RUN if ls /tmp/wheels/fake_int4_quant_cuda-*.whl 2>/dev/null | grep -q .; then \
      pip install /tmp/wheels/fake_int4_quant_cuda-*.whl; \
    else \
      cd /root/miles/miles/backends/megatron_utils/kernels/int4_qat && \
      TORCH_CUDA_ARCH_LIST="8.0 9.0" MAX_JOBS=32 pip install . --no-build-isolation; \
    fi

RUN rm -rf /tmp/wheels
