FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system deps
RUN apt-get update && apt-get install -y \
    wget git curl bzip2 ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Accept conda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create minimal conda env with just Python
RUN conda create -n anycam python=3.11 -y && conda clean -afy

WORKDIR /workspace
COPY . .

# Activate env for all following commands
SHELL ["conda", "run", "-n", "anycam", "/bin/bash", "-c"]

# Install PyTorch NIGHTLY for RTX 5090 (sm_120) support
# Install separately to avoid version conflicts between nightly builds
RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
RUN pip install --pre --no-deps torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Install xformers (latest)
RUN pip install xformers

# Install main dependencies (no version pins)
RUN pip install \
    numpy opencv-python pillow scipy matplotlib pandas \
    omegaconf hydra-core tensorboard wandb tqdm \
    einops timm transformers huggingface-hub safetensors \
    kornia scikit-image imageio moviepy \
    pytorch-ignite pycolmap pyceres

# Install project packages
RUN pip install -e /workspace/anycalib

# Install UniDepth with --no-deps to prevent PyTorch downgrade, then install its deps separately
RUN git clone https://github.com/Brummi/UniDepth.git /workspace/UniDepth && \
    cd /workspace/UniDepth && pip install --no-deps -e . && \
    pip install fvcore iopath yacs h5py tables blosc2

# Install pytorch3d from source (not on PyPI)
# Use --no-build-isolation so it can find the already-installed torch
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

RUN pip install -e /workspace/unimatch

CMD ["conda", "run", "-n", "anycam", "bash"]
