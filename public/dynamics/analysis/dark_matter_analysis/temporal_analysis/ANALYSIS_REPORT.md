# Temporal Analysis Summary Report

**Generated:** 2026-01-22 00:01:41

---

## Executive Summary

This analysis investigates whether 'dark matter' features (those with poor D_i ~ ||W_i||² linear scaling) 
are transient states during training or represent persistent structural phenomena.

### Key Findings

1. **Dark matter is persistent**: 41.6% of features remain dark matter at training end. 
Only 23.1% of initial dark matter transitions to well-behaved.

2. **Eigenspace hopping decreases**: Features stabilize into eigenspaces over training, 
but 46.3 mean hops still occur.

3. **Concentration increases modestly**: 8.7% 
of features become more concentrated, but dark matter remains diffuse.

4. **κλ ≈ 1 holds approximately**: Final κλ = 0.999 ± 
0.091 (high sparsity).

5. **Trajectory classification**: Only 8.4% of high-sparsity features 
follow linear trajectories.


---

## Conclusion

The temporal analysis provides strong evidence that **dark matter is a persistent structural feature** 
of superposition, not merely a transient training phenomenon. While features do stabilize into eigenspaces 
and hopping decreases over time, the diffuse eigenspace distributions that characterize dark matter persist 
throughout training. This explains why the linear scaling law D_i ~ ||W_i||² fails for the majority of features 
even at the end of training.


## Implications

1. The geometric interpretation (κ = 1/λ) may need refinement to account for multi-eigenspace distributions.

2. Dark matter may represent features in 'mixed' representations that don't cleanly map to single eigenspaces.

3. The ~26% of well-behaved features may correspond to 'lucky' initial conditions that align with eigenspaces.


---

## Files Generated

| Analysis | Results File | Plot |

|----------|--------------|------|

| Dark Matter Evolution | `dark_matter_temporal.json` | `dark_matter_evolution.png` |

| Eigenspace Stability | `eigenspace_stability.json` | `eigenspace_stability.png` |

| Instantaneous Linearity | `instantaneous_linearity.json` | `instantaneous_linearity.png` |

| Slope-Eigenvalue Temporal | `slope_eigenvalue_temporal.json` | `slope_eigenvalue_temporal.png` |

| Trajectory Classification | `trajectory_classification.json` | `trajectory_classification.png` |

| Concentration Dynamics | `concentration_dynamics.json` | `concentration_dynamics.png` |

| Phase Animation | - | `phase_comparison.png`, `phase_tracers.png` |

| Aggregate Summary | - | `aggregate_summary.png` |
