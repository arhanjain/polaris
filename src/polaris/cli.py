"""
Polaris command-line entry point.

Currently supports:
  polaris upload <env_dir> [--options]
"""

from __future__ import annotations

import sys

from polaris import hf_upload


def main(argv: list[str] | None = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        print("\nUsage: polaris upload <env_dir> [--repo-id ...]")
        sys.exit(0)

    command, *rest = args

    if command == "upload":
        hf_upload.main(rest)
        return

    print(f"Unknown command: {command}")
    print("Supported commands: upload")
    sys.exit(1)


if __name__ == "__main__":
    main()

