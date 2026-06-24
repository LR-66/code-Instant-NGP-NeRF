"""Main entry point for the framework."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="NeRF Teaching Scene Reconstruction")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train_nerf", "distill", "vr", "evaluate"],
                        help="Run mode")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (optional)")
    args = parser.parse_args()

    if args.mode == "train_nerf":
        from train_nerf import main as train_main
        train_main()
    elif args.mode == "distill":
        from distill_gaussian import main as distill_main
        distill_main()
    elif args.mode == "vr":
        from vr_interact import main as vr_main
        vr_main()
    elif args.mode == "evaluate":
        from run_evaluation import main as eval_main
        eval_main()

if __name__ == "__main__":
    main()