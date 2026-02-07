# Experiment Journal

This document tracks experiments, observations, and decisions made during model development. Treat this like a lab notebook—write down what you tried, what worked, and what didn't.

---

## Project: FashionMNIST Classification

**Goal**: Train a neural network to classify 28×28 grayscale images into 10 clothing categories.

**Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Baseline Experiment

**Date**: [YYYY-MM-DD]

**Config**: `python main.py --preset default`
- Hidden dims: [256, 128, 64]
- Activation: GELU
- Dropout: 0.1
- Learning rate: 1e-3
- Epochs: 20

**Results**:
- Final train accuracy: ___
- Final val accuracy: ___
- Test accuracy: ___

**Observations**:
- [Did it converge?]
- [Any signs of overfitting?]
- [Which classes are hardest?]

---

## Experiment Template

### Experiment: [Name]

**Date**: [YYYY-MM-DD]

**Hypothesis**: [What are you testing? Why?]

**Changes from baseline**:
```bash
python main.py --[your flags here]
```

**Results**:
| Metric | Baseline | This Run |
|--------|----------|----------|
| Val Acc | ___ | ___ |
| Test Acc | ___ | ___ |
| Train Time | ___ | ___ |

**Observations**:
- [What happened?]
- [Did the hypothesis hold?]

**Conclusions/Next Steps**:
- [What did you learn?]
- [What to try next?]

--

## References

- FashionMNIST paper: Xiao, Rasul, Vollgraf (2017)
- Dataset: https://github.com/zalandoresearch/fashion-mnist