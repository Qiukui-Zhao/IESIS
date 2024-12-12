import os
import subprocess

def run_command(command: str, description: str = ""):
    """Runs a shell command with optional logging."""
    try:
        print(f"Running: {description or command}")
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e.stderr}")

# Install core dependencies
dependencies = [
    "torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html",
    "-U openmim",
    "mmengine",
    "mmcv==2.0.0",
    "opencv-python pillow matplotlib seaborn tqdm pytorch-lightning 'mmdet>=3.1.0' -i https://pypi.tuna.tsinghua.edu.cn/simple",
]

for dep in dependencies:
    run_command(f"pip install {dep}", f"Installing {dep}")

# Clone MMSegmentation repository
repo_url = "https://github.com/open-mmlab/mmsegmentation.git"
repo_version = "v1.1.2"
run_command(f"git clone {repo_url} -b {repo_version}", "Cloning MMSegmentation repository")

# Change to the MMSegmentation directory and install as an editable package
os.chdir("mmsegmentation")
run_command("pip install -v -e .", "Installing MMSegmentation in editable mode")

# Create necessary directories
directories = ["checkpoint", "outputs", "data", "FIGURE", "Configs"]
for dir_name in directories:
    os.makedirs(dir_name, exist_ok=True)
    print(f"Directory created: {dir_name}")

# Verify environment setup
print("\nEnvironment Verification:")
try:
    import torch
    print(f"Pytorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    import mmcv
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print(f"MMCV version: {mmcv.__version__}")
    print(f"CUDA version: {get_compiling_cuda_version()}")
    print(f"Compiler version: {get_compiler_version()}")

    import mmseg
    print(f"MMSegmentation version: {mmseg.__version__}")

except ImportError as e:
    print(f"Import Error: {e}")
