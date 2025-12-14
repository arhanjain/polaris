## Coming Soon :(




## Uploading Environments to HuggingFace

Share your custom environments with the community by uploading them to the [PolaRiS-Evals/PolaRiS-Hub](https://huggingface.co/datasets/PolaRiS-Evals/PolaRiS-Hub) dataset. Uploads are submitted as pull requests for review.

### Environment Structure

Your environment folder must contain:
```
my_environment/
├── assets/
│   ├── object_1/
│   │   └── mesh.usdz
│   ├── object_2/
│   │   └── mesh.usdz
│   └── scene_splat/
│       ├── config.yaml
│       └── splat.ply
├── scene.usd              # Main USD stage file
└── initial_conditions.json
```

### Upload Commands

```bash
# Install the package (adds polaris CLI to PATH)
pip install -e .

# Dry-run validation only (no upload)
polaris upload ./PolaRiS-environments/my_environment --dry-run

# Upload and create a PR (uses HF_TOKEN env var)
export HF_TOKEN=your_huggingface_write_token
polaris upload ./PolaRiS-environments/my_environment \
  --pr-title "Add my_environment" \
  --pr-description "Description of the environment"

# Upload to a different repo
polaris upload ./PolaRiS-environments/my_environment \
  --repo-id your-org/your-dataset \
  --pr-title "Add my_environment"
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Validate only, don't upload |
| `--pr-title` | Title for the pull request |
| `--pr-description` | Description/body for the PR |
| `--repo-id` | Target HF dataset (default: `PolaRiS-Evals/PolaRiS-Hub`) |
| `--branch` | Target branch (default: `main`) |
| `--token` | HF token (or use `HF_TOKEN` env var) |
| `--strict` | Treat validation warnings as errors |
| `--require-pxr` | Fail if USD files can't be opened (requires pxr) |
| `--skip-validation` | Skip validation (not recommended) |

### Managing Your PR Locally

After creating a PR, you can check it out locally to make changes:

```bash
# Clone the dataset repo
git clone https://huggingface.co/datasets/PolaRiS-Evals/PolaRiS-Hub
cd PolaRiS-Hub

# Fetch and checkout PR (replace <PR_NUMBER> with your PR number)
git fetch origin refs/pr/<PR_NUMBER>:pr/<PR_NUMBER>
git checkout pr/<PR_NUMBER>

# Make edits, then push back
git add .
git commit -m "Update environment"
git push origin pr/<PR_NUMBER>:refs/pr/<PR_NUMBER>
```