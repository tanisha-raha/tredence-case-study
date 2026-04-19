# Self-Pruning Neural Network
**Tredence Analytics — AI Engineer Case Study**

## What this is
A neural network that learns to prune its own weights during training using learnable gates, temperature annealing, and L1 sparsity regularisation.

## Key features
- Custom `PrunableLinear` layer with learnable gate scores
- Temperature-annealed sigmoid for progressive gate commitment
- Straight-Through Estimator (STE) for hard binary gates at inference
- FLOPs reduction metric alongside sparsity %
- Trains on CIFAR-10 across 3 lambda values (low / medium / high pruning)

## Setup
```bash
pip install torch torchvision matplotlib numpy
```

## Run
```bash
python self_pruning_network.py
```

Results are saved to `./results/` — includes gate distribution plots, training curves, and a summary JSON.
