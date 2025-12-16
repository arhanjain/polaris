#!/usr/bin/env python3
"""
Upload a PolaRiS environment folder to the Hugging Face dataset repository
`PolaRiS-Evals/PolaRiS-Hub` after performing local validation.
Validation tries to catch the most common mistakes (missing assets, malformed
`initial_conditions.json`, unreadable USD), but it cannot guarantee runtime
success inside Isaac Sim. Use this as a fast client-side gate before pushing.
# Example commands:
#   Dry-run validation only:
#   python scripts/upload_env_to_hf.py ./PolaRiS-environments/food_bussing --dry-run
#
#   Upload after validation (uses HF_TOKEN env var):
#   HF_TOKEN=your_token_here python scripts/upload_env_to_hf.py ./PolaRiS-environments/food_bussing
#
#   Upload with explicit token and strict mode:
#   python scripts/upload_env_to_hf.py ./PolaRiS-environments/food_bussing --token your_token_here --strict
#
#   Preferred CLI form (after `pip install -e .`):
#   polaris upload ./PolaRiS-environments/food_bussing --dry-run
#
#   Create a pull request instead of direct commit:
#   polaris upload ./PolaRiS-environments/food_bussing --pr-title "Add food bussing env"
#
#   Target a different HF dataset repo (example: owhan/PolaRiS-environments):
#   ./.venv/bin/python -m polaris.cli upload ./PolaRiS-environments/block_stack_kitchen_v2/ --repo-id owhan/PolaRiS-environments --pr-title "Add block stack kitchen env"
"""

from polaris.hf_upload import main  # type: ignore


if __name__ == "__main__":
    main()
