import sys
import torch
import os
from pathlib import Path
import subprocess

def validate_environment():
    # Environment Validation
    env_checks = {
        "Python Version": sys.version.startswith("3.10"),
        "PyTorch Version": torch.__version__ == "2.5.1",
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda == "12.1",
        "GPU Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "Development Path": os.path.exists("C:/Users/pvmal/anaconda3/envs/ai_dev")
    }

    # IDE and Git Configuration
    ide_checks = {
        "VS Code Extensions": os.path.exists(str(Path.home() / ".vscode/extensions")),
        ".gitignore": os.path.exists(".gitignore"),
        "Git Repository": os.path.exists(".git")
    }

    # Check installed packages using pip
    def check_package(package):
        try:
            subprocess.check_output([sys.executable, '-m', 'pip', 'show', package])
            return True
        except subprocess.CalledProcessError:
            return False

    package_checks = {
        "black": check_package("black"),
        "flake8": check_package("flake8"),
        "pylint": check_package("pylint"),
        "pre-commit": check_package("pre-commit")
    }

    print("=== Week 1, Day 1 Morning Setup Validation ===\n")

    for category, checks in [
        ("Environment Setup", env_checks),
        ("IDE Configuration", ide_checks),
        ("Code Quality Tools", package_checks)
    ]:
        print(f"\n{category}:")
        for check, status in checks.items():
            print(f"✓ {check}: {'Pass' if status else 'Not Found'}")

    # Test CUDA functionality
    if torch.cuda.is_available():
        test_tensor = torch.randn(1000, 1000).cuda()
        torch.matmul(test_tensor, test_tensor)
        print("\nCUDA Test: GPU Matrix Multiplication Successful")

validate_environment()
