# Use Yotta's official base image (Python 3.11, CUDA 12.8, includes /start.sh for SSH etc.)
FROM yottalabsai/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-2025081902

# Switch to root so pip installs to system site-packages (not ~/.local which Yotta volume-mounts over)
USER root

# Install prebuilt wheels + all deps, then re-pin torch and clean up in one layer
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
    https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
    "flash-linear-attention>=0.4.2" \
    && pip install --no-cache-dir \
    "accelerate>=1.13.0" \
    "datasets>=4.8.4" \
    "deepspeed>=0.18.8" \
    "transformers>=5.3.0" \
    ninja packaging \
    && pip install --no-cache-dir \
    tensorboard zstandard pandas pyarrow huggingface_hub \
    jsonargparse[signatures] tokenizers sentencepiece wandb torchmetrics \
    einops braceexpand smart_open opt_einsum cbor2 \
    isort pytest mypy mosaicml-streaming lm-eval==0.4.1 \
    && pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip uninstall -y torchvision torchaudio \
    && rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Clone IBSSM, copy ibm2 into fla/models, and register it in fla's __init__.py
RUN git clone https://github.com/zhuconv/IBSSM.git /opt/IBSSM \
    && cp -r /opt/IBSSM/ibm2_source /usr/local/lib/python3.11/dist-packages/fla/models/ibm2 \
    && sed -i '1i from fla.models.ibm2 import IBM2Config, IBM2ForCausalLM, IBM2Model' \
       /usr/local/lib/python3.11/dist-packages/fla/models/__init__.py

# Install Claude CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# Set HF_TOKEN
# Switch back to the default user
USER ubuntu
