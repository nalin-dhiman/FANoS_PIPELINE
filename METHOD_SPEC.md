\
METHOD SPEC: FANoS (Friction-Adaptive Nosé–Hoover Symplectic Momentum)
===============================================================

This document defines the discrete update implemented in `fanos/fanos_optimizer.py`.
It is written as an implementation-audit spec, not as a convergence guarantee.

Notation
--------
Let θ ∈ R^d be parameters and L(θ) the objective.
Let g_k = ∇L(θ_k) be the gradient at step k.
Let h > 0 be the integrator step size (called `lr` in code).

Diagonal mass (RMS preconditioner)
----------------------------------
We maintain an exponential moving average of squared gradients:

  s_{k} = β s_{k-1} + (1-β) (g_k ⊙ g_k)

and define a diagonal "mass" (or preconditioner denominator)

  m_k = sqrt(s_k) + ε

Identity-mass variant sets m_k ≡ 1.

Thermostatted heavy-ball dynamics (discrete)
-------------------------------------------
We maintain a velocity v_k (same shape as θ). The update is:

Velocity (semi-implicit / symplectic-Euler–style):
  v_{k+1} = (1 - h ζ_k) v_k - h ( g_k ⊘ m_k )

Position:
  θ_{k+1} = θ_k + h v_{k+1}

**Explicit-Euler ablation** replaces the position update with:
  θ_{k+1} = θ_k + h v_k

Kinetic-energy proxy and thermostat
-----------------------------------
Define an instantaneous "temperature" proxy (per group):

  T_inst(k+1) = (1/d) Σ_i [ m_{k,i} * v_{k+1,i}^2 ]

We smooth it:
  T_ema(k+1) = ρ_T T_ema(k) + (1-ρ_T) T_inst(k+1)

We use an annealed target temperature schedule:
  T0(k) = T_min + (T_max - T_min) exp(-k/τ)

Thermostat (Nosé–Hoover style, discretized):
  ζ_{k+1} = clip( ζ_k + (h/Q) ( T_ema(k+1) - T0(k) ), [-ζ_max, +ζ_max] )

**Fixed-friction ablation** sets ζ_k ≡ ζ_const (no thermostat update).

Important caveats (for honest reporting)
----------------------------------------
- The term "symplectic" is used in the numerical-analysis sense for the semi-implicit update on
  Hamiltonian systems. Once friction, clipping, and time-varying diagonal mass are introduced,
  the method is not strictly symplectic. In the paper, we refer to it as "semi-implicit / structure-preserving"
  where appropriate.

- Because θ update includes h v_{k+1} and v update includes h g_k, the effective gradient scale in θ
  is O(h^2) under this parametrization. Hence h should be interpreted as an integrator step size.

- This method is not presented as a general-purpose replacement for AdamW. The pipeline is designed
  to support conservative, evidence-based claims on stiff objectives.

