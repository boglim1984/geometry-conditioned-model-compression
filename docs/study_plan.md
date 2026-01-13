# Study Plan: Geometry-Conditioned Model Compression

## Research Question

Can statistical geometry (covariance structure, effective rank) be exploited to improve model compression, even though it is not intrinsic manifold structure?

## Background

The [functional-geometry falsification study](https://github.com/boglim1984/functional-geometry-hebbian-manifold) demonstrated that apparent low-dimensional structure in neural activations is a **covariance artifact**, not an intrinsic manifold. Whitening destroys geometric structure while preserving performance.

**This study asks**: Even if geometry is statistical, can we use it for compression?

## Hypotheses

### H1: Variance-Weighted Pruning

**Claim**: Pruning decisions weighted by variance contribution will outperform magnitude-based pruning.

**Rationale**: If high-variance directions encode more functional information, removing low-variance components should preserve performance better than magnitude-based heuristics.

**Test**: Compare variance-weighted channel pruning vs. L1 magnitude pruning at matched sparsity (50%, 70%, 90%).

**Falsification**: No improvement over magnitude pruning, or advantage disappears under whitening.

---

### H2: Spectral-Adaptive LoRA

**Claim**: LoRA rank allocation conditioned on layer-wise effective rank will outperform uniform allocation.

**Rationale**: Layers with higher effective rank may benefit from higher-rank adaptation.

**Test**: Compare spectral-adaptive LoRA vs. standard LoRA at matched total parameter budgets.

**Falsification**: No improvement, or advantage disappears when effective rank is artificially equalized.

---

### H3: Whitening Diagnostic

**Claim**: Whitening will reveal whether advantages are statistical or functional.

**Test**: Apply whitening transformations to activations before compression. If geometry-conditioned methods still outperform baselines, the advantage is functional. If not, it was purely statistical.

**Outcome**: This is a diagnostic, not a hypothesis. It classifies the nature of any observed advantages.

---

## Experimental Design

### Phase 1: Variance-Weighted Pruning

**Model**: ResNet-18, CIFAR-10  
**Baseline**: L1 structured pruning (magnitude-based channel removal)  
**Method**: Variance-weighted channel pruning  
**Sparsity Levels**: 50%, 70%, 90%  
**Metrics**: Top-1 accuracy, FLOPs, parameter count  
**Runs**: 5 random seeds per configuration  

**Procedure**:
1. Train baseline ResNet-18 to convergence
2. Compute per-channel variance contributions
3. Prune channels by:
   - **Baseline**: Lowest L1 norm
   - **Ours**: Lowest variance contribution
4. Fine-tune pruned models (10 epochs)
5. Compare accuracy at matched sparsity

**Whitening Control**:
- Apply ZCA whitening to activations before computing variance
- Re-run pruning with whitened statistics
- Compare performance degradation

---

### Phase 2: Spectral-Adaptive LoRA

**Model**: GPT-2 Small, WikiText-103  
**Baseline**: Standard LoRA (uniform rank r=8 across all layers)  
**Method**: Spectral-adaptive LoRA (rank ∝ effective rank)  
**Parameter Budget**: Matched total parameters  
**Metrics**: Perplexity, parameter count  
**Runs**: 3 random seeds per configuration  

**Procedure**:
1. Fine-tune GPT-2 with standard LoRA (r=8)
2. Compute layer-wise effective rank (participation ratio)
3. Allocate LoRA rank proportional to effective rank
4. Fine-tune with spectral-adaptive LoRA
5. Compare perplexity at matched parameter budgets

**Whitening Control**:
- Apply layer-wise whitening before computing effective rank
- Re-allocate ranks with whitened statistics
- Compare performance degradation

---

## Success Criteria

### Quantitative

- **Accuracy Improvement**: ≥2% over baseline at matched budgets
- **Statistical Significance**: p < 0.05 (paired t-test, 5 runs)
- **Computational Overhead**: <10% additional training time

### Qualitative

- Methods are **reproducible** (clear implementation, public code)
- Results are **interpretable** (clear attribution of improvements)
- Findings are **falsifiable** (whitening diagnostic provides clear test)

---

## Falsification Criteria

### Strong Falsification

- **No improvement over baselines** at any sparsity/budget level
- **Whitening destroys all advantages** → geometry was purely statistical, not functional

### Weak Falsification

- Improvements are **not statistically significant**
- Improvements **disappear at scale** (larger models, datasets)
- Methods require **prohibitive computational overhead**

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Variance-Weighted Pruning | 2 weeks | Notebook + results |
| Phase 2: Spectral-Adaptive LoRA | 2 weeks | Notebook + results |
| Whitening Diagnostics | 1 week | Analysis + interpretation |
| Documentation | 1 week | Final report |

**Total**: ~6 weeks

---

## Expected Outcomes

### Scenario 1: Strong Success

- Geometry-conditioned methods outperform baselines
- Advantages **survive whitening** → functional signal
- **Conclusion**: Statistical geometry encodes compressible functional structure

### Scenario 2: Weak Success

- Geometry-conditioned methods outperform baselines
- Advantages **disappear under whitening** → statistical artifact
- **Conclusion**: Geometry is useful but not fundamental

### Scenario 3: Falsification

- No improvement over baselines
- **Conclusion**: Statistical geometry does not encode compressible structure

---

## Open Questions

1. **Generalization**: Do advantages transfer across architectures/datasets?
2. **Scaling**: Do advantages persist at larger model scales?
3. **Mechanism**: What functional properties correlate with variance/effective rank?

---

## References

- [Functional Geometry Falsification Study](https://github.com/boglim1984/functional-geometry-hebbian-manifold)
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
- Li et al. (2016). Pruning Filters for Efficient ConvNets
- Pope et al. (2021). The Intrinsic Dimension of Images and Its Impact on Learning
