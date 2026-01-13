# Geometry-Conditioned Model Compression

## Overview

This repository investigates whether **activation-space anisotropy** (covariance structure, effective rank) can be exploited to improve model compression and parameter-efficient fine-tuning beyond magnitude-based heuristics.

### Foundational Position

**We do NOT assume topological manifolds.**  
**We do NOT assume whitening-invariant structure.**  
**We treat geometry as variance-encoded functional signal.**

Geometry in this work refers to **statistical structure**: covariance, anisotropy, and effective rank of activation distributions. These are measurable properties that may or may not carry functional information. Whitening serves as a falsification control—if geometry-conditioned methods fail under whitening, the signal was statistical, not topological.

## Key Question

**Can geometry-conditioned methods outperform SOTA compression when constrained to identical parameter budgets?**

Specifically:
- Can variance-weighted pruning outperform magnitude-based pruning?
- Can spectral-adaptive LoRA outperform standard LoRA?
- Do these advantages survive whitening transformations?

## Relationship to Prior Work

This repository is a **constructive follow-up** to the falsification study:  
[**Functional Geometry in Deep Neural Networks**](https://github.com/boglim1984/functional-geometry-hebbian-manifold)

That study demonstrated:
- Apparent low-dimensional structure in neural activations is **covariance artifact**, not intrinsic manifold
- Whitening destroys geometric structure while preserving classification performance
- Hebbian alignment is a **variance-matching phenomenon**, not manifold discovery

This repository asks: **Given that geometry is statistical structure, can we still exploit it for compression?**

## Experimental Approach

### Hypotheses

**H1: Variance-Weighted Pruning**  
Pruning neurons/channels weighted by their contribution to output variance will outperform magnitude-based pruning at matched sparsity levels.

**H2: Spectral-Adaptive LoRA**  
LoRA rank allocation conditioned on layer-wise effective rank will outperform uniform rank allocation at matched parameter budgets.

**H3: Whitening Diagnostic**  
If geometry-conditioned methods fail under whitening, the advantage was statistical structure. If they survive, the advantage is functional.

### Baselines

- **L1 Structured Pruning**: Magnitude-based channel/neuron removal
- **Standard LoRA**: Uniform rank allocation across layers

### Geometry-Conditioned Variants

- **Variance-Weighted Pruning**: Weight pruning decisions by variance contribution
- **Spectral-Adaptive LoRA**: Allocate rank proportional to effective rank

### Success Criteria

- Geometry-conditioned methods achieve **≥2% accuracy improvement** over baselines at matched budgets
- Improvements are **statistically significant** (p < 0.05, multiple runs)
- Methods are **computationally practical** (no prohibitive overhead)

### Falsification Criteria

- Whitening destroys all advantages → geometry was purely statistical
- No improvement over baselines → geometry does not encode compressible structure
- Improvements disappear at scale → method is dataset/architecture-specific artifact

## Repository Structure

```
geometry-conditioned-model-compression/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── docs/
│   └── study_plan.md                     # Detailed experimental plan
└── notebooks/
    └── 01_variance_weighted_pruning.ipynb # Initial pruning experiments
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Experiments are organized as Jupyter notebooks in `notebooks/`. Each notebook is self-contained and documents its own hypotheses, methods, and results.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{geometry-conditioned-compression-2026,
  author = {O'Flaherty, Sean},
  title = {Geometry-Conditioned Model Compression},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/boglim1984/geometry-conditioned-model-compression}
}
```

## Contact

For questions or collaboration: [GitHub Issues](https://github.com/boglim1984/geometry-conditioned-model-compression/issues)
