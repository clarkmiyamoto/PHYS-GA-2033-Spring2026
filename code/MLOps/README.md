All of the previous tutorials were done in Jupyter Notebooks, I want to give an example written as a python script.

> 🚧 Warning:
> 1. I am not an ML engineer so take this with a grain of salt.
> 2. I used Claude to make part of this.

This is assuming you already have a predefined methodology &

# Layout
# ML for Physics: Example Codebase

A pedagogical codebase demonstrating how to structure a machine learning project for physics applications. Uses **2D Ising model energy prediction** as an example problem.

## Project Structure

```
example_project/
├── config.py      # Experiment configuration (dataclasses)
├── data.py        # Data generation and loading
├── model.py       # Neural network architectures
├── trainer.py     # Training loop and optimization
├── logger.py      # WandB and local logging
├── utils.py       # Helper functions (seeding, device, analysis)
├── main.py        # Entry point for experiments
├── journal.md     # Experiment tracking (lab notebook)
└── requirements.txt
```

## Quick Start

```bash
# Run with default settings
python main.py

# Quick debug run (small model, few samples)
python main.py --preset debug

# Custom configuration
python main.py --lr 0.0005 --epochs 200 --hidden 256 128 64
```

## Configuration System

Experiments are fully specified by `ExperimentConfig`, which contains:
- `ModelConfig`: Architecture (hidden dims, activation, dropout)
- `TrainConfig`: Optimization (learning rate, epochs, scheduler)
- `DataConfig`: Dataset (lattice size, number of samples, temperature range)
- `LogConfig`: Logging (WandB, checkpointing)

Use presets for quick experiments:
```bash
python main.py --preset debug   # Fast iteration
python main.py --preset small   # Quick training
python main.py --preset default # Standard run
python main.py --preset large   # High-quality training
```

## Command Line Options

```
Model:
  --hidden 128 64 32    Hidden layer dimensions
  --activation gelu     Activation function
  --dropout 0.1         Dropout rate
  --architecture mlp    Architecture type (mlp, residual_mlp)

Training:
  --lr 0.001           Learning rate
  --epochs 100         Number of epochs
  --batch-size 64      Batch size
  --optimizer adamw    Optimizer (adam, adamw, sgd)
  --scheduler cosine   LR scheduler (none, cosine, step)

Data:
  --lattice-size 8     Ising lattice size L (L×L)
  --num-train 10000    Training samples
  --num-val 2000       Validation samples
  --num-test 2000      Test samples

Logging:
  --wandb              Enable Weights & Biases
  --run-name myrun     Name for this run
  --no-checkpoints     Disable saving checkpoints

Other:
  --seed 42            Random seed
  --device auto        Device (auto, cuda, mps, cpu)
  --config config.json Load configuration from file
```

## Tracking Experiments

1. **Config files**: Saved automatically to `checkpoints/{run_name}/config.json`
2. **Metrics**: Saved to `checkpoints/{run_name}/epoch_metrics.json`
3. **Journal**: Manually track observations in `journal.md`
4. **WandB**: Enable with `--wandb` for cloud experiment tracking

## Extensions for Students

### Easy
- [ ] Plot training curves from saved metrics
- [ ] Add early stopping based on validation loss
- [ ] Try different optimizers (SGD with momentum, AdamW)

### Medium
- [ ] Implement spin-flip symmetry in the model (`SymmetryAwareMLP`)
- [ ] Add temperature as an input feature
- [ ] Train on one lattice size, test on another (generalization)

### Harder
- [ ] Implement a CNN that exploits spatial structure
- [ ] Phase classification: predict if T < Tc or T > Tc
- [ ] Critical temperature estimation from finite-size scaling

## Key Concepts Demonstrated

| ML Concept | Physics Analogy |
|------------|-----------------|
| Loss function | Energy functional to minimize |
| Gradient descent | Steepest descent dynamics |
| Learning rate | Temperature in annealing |
| Batch size | Ensemble size for averaging |
| Regularization | Prior constraints on solutions |
| Early stopping | Avoiding equilibration to noise |
| Residual connections | Perturbation theory (small corrections) |
| Dropout | Disorder averaging |

## Tips for Your Own Projects

1. **Start simple**: Get a basic model working before adding complexity
2. **Track everything**: Use `journal.md` like a lab notebook
3. **Version configs, not just code**: Configs determine experiments
4. **Validate incrementally**: Check each component works before combining
5. **Visualize often**: Plot predictions, errors, training curves

## License

MIT License - use freely for educational purposes.
