# pipeline.py
import subprocess
import sys

def run_step(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_step([sys.executable, "preprocess.py"])
    run_step([sys.executable, "train.py"])
    run_step([sys.executable, "predict.py"])
    print("\nDone.")
