from __future__ import annotations

import argparse

from src.training.checkpoints import average_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    average_checkpoints(args.checkpoints, args.output)


if __name__ == "__main__":
    main()
