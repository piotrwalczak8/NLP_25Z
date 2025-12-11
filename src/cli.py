# src/cli.py
import argparse
from pathlib import Path
from src.runner import run  

def parse_args():
    p = argparse.ArgumentParser(description="Run sentiment experiments")
    p.add_argument("-c", "--config", default="config/config_IMDB.json",
                   help="Path to config JSON (default: config/config_IMDB.json)")
    p.add_argument("--only", choices=["svm","bert","llama","all"], default="all",
                   help="Run only selected model(s)")
    p.add_argument("--dry-run", action="store_true", help="Don't train; just print planned actions")
    p.add_argument("--seed", type=int, default=None, help="Optional random seed override")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    # opts przekazywane do runnera — runner może je zignorować jeśli nie obsługuje
    opts = {
        "only": args.only,
        "dry_run": args.dry_run,
        "seed": args.seed
    }
    run(str(cfg_path), opts)
