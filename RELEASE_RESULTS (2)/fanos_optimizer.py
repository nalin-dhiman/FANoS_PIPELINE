\
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import math
import torch


@dataclass
class TemperatureSchedule:
    """Exponential annealing T0(k) = Tmin + (Tmax-Tmin)*exp(-k/tau)."""
    Tmax: float = 1e-3
    Tmin: float = 0.0
    tau: float = 20000.0

    def __call__(self, step: int) -> float:
        if self.tau <= 0:
            return float(self.Tmin)
        return float(self.Tmin + (self.Tmax - self.Tmin) * math.exp(-step / self.tau))


class FANoS(torch.optim.Optimizer):
    """
    FANoS: Friction-Adaptive Nosé–Hoover Semi-Implicit Momentum (PyTorch)

    Discrete update (per parameter element i), with step size h = lr:

        s_i <- beta*s_i + (1-beta)*g_i^2
        m_i <- sqrt(s_i) + eps               (identity-mass variant uses m_i = 1)

        v_i <- (1 - h*zeta) * v_i - h * (g_i / m_i)

        theta_i <- theta_i + h*v_i           (semi-implicit / symplectic-Euler–style)

    Thermostat (per parameter-group scalar zeta):
        T_inst <- mean_i( m_i * v_i^2 )
        T_ema  <- rho_T*T_ema + (1-rho_T)*T_inst
        zeta   <- clip(zeta + (h/Q)*(T_ema - T0(step)), [-zeta_clip, +zeta_clip])

    Ablations:
      - explicit_euler=True uses theta <- theta + h*v_old
      - fixed_friction=True sets zeta <- zeta_const (no thermostat update)
      - identity_mass=True uses m_i ≡ 1 (no RMS state)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.999,
        eps: float = 1e-8,
        Q: float = 1.0,
        T0_max: float = 1e-3,
        T0_min: float = 0.0,
        tau: float = 20000.0,
        rho_T: float = 0.9,
        zeta_clip: float = 10.0,
        grad_clip: Optional[float] = 1.0,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        # ablations / variants
        explicit_euler: bool = False,
        fixed_friction: bool = False,
        zeta_const: float = 0.0,
        identity_mass: bool = False,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if Q <= 0:
            raise ValueError("Q must be positive")
        if not (0.0 <= rho_T < 1.0):
            raise ValueError("rho_T must be in [0, 1)")
        if zeta_clip < 0:
            raise ValueError("zeta_clip must be >= 0 (0 disables clipping)")

        defaults = dict(
            lr=lr, beta=beta, eps=eps, Q=Q,
            T0_max=T0_max, T0_min=T0_min, tau=tau,
            rho_T=rho_T, zeta_clip=zeta_clip,
            grad_clip=grad_clip, weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            explicit_euler=explicit_euler,
            fixed_friction=fixed_friction,
            zeta_const=zeta_const,
            identity_mass=identity_mass,
        )
        super().__init__(params, defaults)

        self.schedule = TemperatureSchedule(Tmax=T0_max, Tmin=T0_min, tau=tau)
        self._step_count = 0

        # Group-level thermostat state
        for group in self.param_groups:
            group.setdefault("zeta", 0.0)
            group.setdefault("T_ema", float(T0_max))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        step = self._step_count
        T0 = self.schedule(step)

        for group in self.param_groups:
            h = float(group["lr"])
            beta = float(group["beta"])
            eps = float(group["eps"])
            Q = float(group["Q"])
            rho_T = float(group["rho_T"])
            zeta_clip = float(group["zeta_clip"])
            grad_clip = group.get("grad_clip", None)
            weight_decay = float(group.get("weight_decay", 0.0))
            decoupled_wd = bool(group.get("decoupled_weight_decay", True))

            explicit_euler = bool(group.get("explicit_euler", False))
            fixed_friction = bool(group.get("fixed_friction", False))
            zeta_const = float(group.get("zeta_const", 0.0))
            identity_mass = bool(group.get("identity_mass", False))

            zeta = float(group.get("zeta", 0.0))
            T_ema = float(group.get("T_ema", T0))

            # ---- optional global grad clip for this group ----
            if grad_clip is not None:
                # compute norm over tensors in group
                sq = 0.0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    if g.is_sparse:
                        raise RuntimeError("FANoS does not support sparse gradients")
                    sq += float(g.pow(2).sum().item())
                gnorm = math.sqrt(sq) if sq > 0 else 0.0
                clip_scale = min(1.0, float(grad_clip) / (gnorm + 1e-12))
            else:
                clip_scale = 1.0

            # ---- per-step temperature accumulation (group) ----
            temp_sum = 0.0
            temp_count = 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("FANoS does not support sparse gradients")
                if not torch.isfinite(g).all():
                    continue

                # weight decay
                if weight_decay != 0.0:
                    if decoupled_wd:
                        p.data.mul_(1.0 - h * weight_decay)
                    else:
                        g = g.add(p.data, alpha=weight_decay)

                if clip_scale != 1.0:
                    g = g.mul(clip_scale)

                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p.data)
                    if not identity_mass:
                        state["s"] = torch.zeros_like(p.data)

                v = state["v"]

                # mass / preconditioner
                if identity_mass:
                    m = None  # m ≡ 1
                    g_eff = g
                else:
                    s = state["s"]
                    s.mul_(beta).addcmul_(g, g, value=1.0 - beta)
                    m = s.sqrt().add(eps)
                    g_eff = g / m

                # velocity update: v_{k+1} = (1 - h zeta) v_k - h g_eff
                v_old = v.clone() if explicit_euler else None
                v.mul_(1.0 - h * zeta).add_(g_eff, alpha=-h)

                # position update
                if explicit_euler:
                    p.data.add_(v_old, alpha=h)  # theta += h v_k
                else:
                    p.data.add_(v, alpha=h)      # theta += h v_{k+1}

                # temperature contribution: mean(m * v^2) or mean(v^2) for identity mass
                if identity_mass:
                    temp_sum += float((v * v).sum().item())
                else:
                    temp_sum += float((m * v * v).sum().item())
                temp_count += v.numel()

            # thermostat update (group-level)
            if temp_count > 0:
                T_inst = temp_sum / float(temp_count)
                T_ema = rho_T * T_ema + (1.0 - rho_T) * T_inst

                if fixed_friction:
                    zeta = zeta_const
                else:
                    zeta = zeta + (h / Q) * (T_ema - T0)
                    if zeta_clip > 0:
                        zeta = max(-zeta_clip, min(zeta_clip, zeta))

            group["zeta"] = float(zeta)
            group["T_ema"] = float(T_ema)

        return loss
