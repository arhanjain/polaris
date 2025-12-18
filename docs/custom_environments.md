# Creating Custom Environments

The environments we provide were scanned using ZED cameras, but the reconstruction pipeline is camera agnostic.

Capture a dense view video of a scene without motion blur, and run it through [COLMAP](https://colmap.github.io/install.html)

Once you have your COLMAP dataset, follow the instructions in [2DGS](https://github.com/hbb1/2d-gaussian-splatting) to obtain a splat and corresponding extracted mesh.

Turn the `fuse_post.ply` mesh into a USD, and create an asset directory that follows this structure.
```
new_asset/
├── mesh.usd
├── splat.ply
├── textures/ (optional, if USD requires textures)
└── config.yaml (optional, USD parameter configuratoin)
```

Using the [online scene composition GUI](https://polaris-evals.github.io/compose-environments/), create a USD stage that composes the objects in the scene. Export and unzip the USD with the command below.
```
unzip scene.zip -d PolaRiS-Hub/new_env/
```

You should now have a directory that looks something like this:
```
PolaRiS-Hub/
└── new_env/
    ├── assets/
    │   ├── object_1/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   ├── object_2/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   └── scene_splat/
    │       ├── config.yaml
    │       └── splat.ply
    ├── scene.usda             # Main USD stage file
    └── initial_conditions.json  (defined via GUI)
```

Add the new environment to the [environments file](../src/polaris/environments/__init__.py), following the same pattern as the default 6 environments. You can also see how to define a rubric to score rollouts with just a few lines of code. Now you can use this environment by changing the `--environment` flag in the eval script.

After testing the environment, please consider submitting a PR to upload it to the [PolaRiS-Hub](https://huggingface.co/datasets/owhan/PolaRiS-Hub)! See below for instructions.

## Uploading Environments to HuggingFace

Share your custom environments with the community by uploading them to the [PolaRiS-Hub](https://huggingface.co/datasets/owhan/PolaRiS-Hub) dataset. **All uploads are automatically submitted as pull requests** (not direct commits) for review and quality control.

### Environment Structure

Your environment folder should look something like this:
```
PolaRiS-Hub/
└── new_env/
    ├── assets/
    │   ├── object_1/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   ├── object_2/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   └── scene_splat/
    │       ├── config.yaml
    │       └── splat.ply
    ├── scene.usda             # Main USD stage file
    └── initial_conditions.json  (defined via GUI)
```

### Upload Commands

```bash
uv run scripts/upload_env_to_hf.py ./PolaRiS-Hub/new_env --pr-title "Add new_env" --pr-description "Description of the environment"
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Validate only, don't upload |
| `--pr-title` | Title for the pull request |
| `--pr-description` | Description/body for the PR |
| `--repo-id` | Target HF dataset (default: `owhan/PolaRiS-Hub`) |
| `--branch` | Target branch (default: `main`) |
| `--token` | HF token (or use `HF_TOKEN` env var) |
| `--strict` | Treat validation warnings as errors |
| `--require-pxr` | Fail if USD files can't be opened (requires pxr) |
| `--skip-validation` | Skip validation (not recommended) |

### How PRs Work for HuggingFace Datasets

When you run `polaris upload`, the tool automatically:

1. Validates your environment structure locally
2. Creates a pull request (not a direct commit) to the target dataset
3. Returns the PR URL or instructions to view it

**Viewing Your PR:**

- After upload, the CLI will print the PR URL (e.g., `https://huggingface.co/datasets/owhan/PolaRiS-Hub/discussions/<PR_NUMBER>`)
- You can also view all PRs at: `https://huggingface.co/datasets/owhan/PolaRiS-Hub/discussions`
- PRs must be reviewed and merged by dataset maintainers before your environment appears in the dataset

**Merging Your PR:**

- Navigate to the PR URL in your browser
- Review the changes in the "Files" tab
- Click "Publish" when ready to merge (requires write access to the dataset)

### Managing Your PR Locally

After creating a PR, you can check it out locally to make changes:

```bash
# Clone the dataset repo
git clone https://huggingface.co/datasets/owhan/PolaRiS-Hub
cd PolaRiS-Hub

# Fetch and checkout PR (replace <PR_NUMBER> with your PR number from the upload output)
git fetch origin refs/pr/<PR_NUMBER>:pr/<PR_NUMBER>
git checkout pr/<PR_NUMBER>

# Make edits, then push back
git add .
git commit -m "Update environment"
git push origin pr/<PR_NUMBER>:refs/pr/<PR_NUMBER>
```
