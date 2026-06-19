# Platformer Lab

> CLI-first AI showcase for a handcrafted 2D platformer — pure NumPy Value MPC, A* baselines, and risk-aware planning.

## Quickstart

```bash
pip install -e .
platformer-lab
```

Requires Python 3.10+ and NumPy. To run without installing:

```bash
PYTHONPATH=src python -m platformer_lab
```

Resume from a saved checkpoint:

```bash
platformer-lab --resume-model outputs/checkpoint/primary.npz
```

Set `PLATFORMER_LAB_OUTPUT_DIR` to redirect generated files elsewhere.

## Pipeline

1. Train a Value MPC controller on procedurally-generated platformer levels
2. Benchmark four controllers on held-out levels — metrics printed to the terminal
3. Run a holdout generalization study, training on a subset of level families and saving a separate checkpoint

## Outputs

```
outputs/
├── checkpoint/
│   ├── primary.npz
│   └── holdout.npz
└── config/
    └── holdout.json
```

## Controllers

| Controller           | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| Static A*            | A* on the static tile grid, ignores moving enemies                   |
| Dynamic A*           | A* re-planned each step with enemy positions as obstacles            |
| Value MPC            | Learned terminal-value model for short-horizon rollout scoring       |
| Risk-Aware Value MPC | Value MPC extended with learned state-risk and action-risk penalties |

## License

MIT. See [LICENSE](LICENSE).
