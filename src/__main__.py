import sys
from pathlib import Path

from .run_experiments import run_experiments


def main() -> None:
    config_path: Path = "config.toml"

    if len(sys.argv) > 1:
        config_path =sys.argv[1]

    run_experiments(config_path)


if __name__ == "__main__":
    main()