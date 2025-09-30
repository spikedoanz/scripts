# ------------------
# os:   ubuntu 24.04
# arch: x86-64
# ------------------

CUDA_VERSION = "13"
CUDA_DRIVER  = "580"

DOTFILES_URL = "https://raw.githubusercontent.com/spikedoanz/dotfiles/main"

bootstrap = [
  "sudo apt update",
  "sudo apt install xz-utils",
  "sudo apt install zsh",
  "sudo apt install curl",
  "sudo apt install gcc"
  # https://nixos.org/download/#nix-install-linux
  "printf 'y\n\ny\n\ny\n\n' | sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon",
  "source ~/.bashrc",
  # python tooling
  "nix-env -iA nixpkgs.uv",
  "nix-env -iA nixpkgs.ruff",
  "sudo chsh -s $(which zsh)",
]

_system_deps    = ["fzf", "eza", "bat", "magic-wormhole", "tmux", "neovim", "git"]
_dev_deps       = ["uv", "ruff", "pyright"]

dependencies = [
  f"nix-env -iA nixpkgs.{dep}" for dep in _system_deps + _dev_deps
]

dotfiles = [
  # zsh    
  f"curl -o ~/.zshrc                        {DOTFILES_URL}/.zshrc_cuda",
  # tmux 
  "mkdir -p ~/.config/tmux",
  "git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm",
  f"curl -o ~/.config/tmux/tmux.conf        {DOTFILES_URL}/tmux/tmux_remote.conf",
  # nvim
  "mkdir -p ~/.config/nvim",
  f"curl -o ~/.config/nvim/init.lua         {DOTFILES_URL}/nvim/init.lua",
]


cuda = [
  f"sudo ubuntu-drivers install nvidia:{CUDA_DRIVER}",
  "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin",
  "sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600",
  "wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb",
  "sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb",
  "sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/",
  "sudo apt-get update",
  "sudo apt-get -y install cuda-toolkit-13-0",
]

extras = [
  "sudo apt install python3.12-dev", # triton needs this sometimes
]

def has_cuda():
  import subprocess
  try:
    return subprocess.run(['nvidia-smi'], capture_output=True, check=True).returncode == 0 
  except Exception: 
    return False

all = bootstrap + dependencies + dotfiles

if not has_cuda(): all += cuda

if __name__ == "__main__":
  for _ in all:
    print(_)
