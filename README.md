# PolaRiS

PolaRiS is a evaluation framework for generalist policies. It provides a simple interface for evaluating models, rendering environments, and running experiments with minimal setup.

## Installation

### Clone the repository (recursively)

```bash
git clone --recursive git@github.com:YOUR_USERNAME/PolaRiS.git
cd PolaRiS
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Sync environment with uv

```bash
uv sync
```

## Usage

Run an evaluation on a USD environment:

```bash
uv run scripts/eval.py --usd /path/to/environment.usd
```

## Project Structure

```text
PolaRiS/
├── scripts/
│   └── eval.py
├── data/
│   └── assets/
│   └── environments/
├── src/polaris/
└── README.md
```


TODO
- If nvcc, cuda toolkit isnt installed, what to do
- supports CUDA 12 only
- make sure the TORCH archirecutre list is correct (mineby default included way more than it needed)
- have correct version of gxx (my versions was too new)
- clear torch_extensions cache in between builds and env changes