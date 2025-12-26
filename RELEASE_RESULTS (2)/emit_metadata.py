
import sys
import torch
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "run_metadata.txt"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("FANoS Pipeline Run Metadata\n")
        f.write("===========================\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
        
        # Git hash
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            f.write(f"Git Commit: {commit}\n")
        except Exception:
            f.write("Git Commit: (git not found or not a repo)\n")
            
        f.write("\nConfiguration\n")
        f.write("-------------\n")
        f.write("Seeds: Fixed per task (see runners)\n")
        f.write("Rosenbrock Budget: 3000 evals\n")
        f.write("Quadratic Budget: 3000 evals\n")
        
    print(f"[metadata] wrote {OUT}")

if __name__ == "__main__":
    main()
