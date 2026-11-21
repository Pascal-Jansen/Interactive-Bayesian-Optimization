# bo_kernels.py — configurable GP covariance presets and hierarchical parameter gating
# for the BoTorch / botorch-rust backends (bo.py and mobo.py).
#
# Standard library only at module import time (like bo_library.py and context_support.py),
# so the stub-only CI suite imports this module without torch, gpytorch or botorch installed.

import dataclasses
import math

DEFAULT_KERNEL_PRESET = "default"

KERNEL_PRESETS = (
    "default",
    "matern52_ard",
    "matern32_ard",
    "matern12_ard",
    "matern32_dimscaled_ard",
    "matern32_iso",
    "rq_iso",
    "shared_add_matern32",
    "shared_add_inter_matern32",
)

# Presets implemented with a custom Python GPyTorch Kernel subclass (1-parameter shared-lengthscale
# 1D additive kernel) rather than a built-in Rust kernel in botorch_rs.
BOTORCH_RS_CUSTOM_PRESETS = (
    "shared_add_matern32",
    "shared_add_inter_matern32",
)


@dataclasses.dataclass(frozen=True)
class ParameterGate:
    """Declarative parent->child hierarchical gate in normalized [0, 1] parameter space.

    When parent parameter u_p <= threshold, the child parameter u_c collapses to ``neutral``
    (c_0 in [0, 1]), eliminating artificial Euclidean kernel distances among designs where the
    child attribute is perceptually or physically inactive:
        u'_c = c_0 + g(u_p) * (u_c - c_0).
    """

    child_index: int
    child_key: str
    parent_index: int
    parent_key: str
    threshold: float = 0.0
    neutral: float = 0.0
    sharpness: float = 0.0

    def to_record(self):
        return {
            "child": self.child_key,
            "gated_by": self.parent_key,
            "threshold": self.threshold,
            "neutral": self.neutral,
            "sharpness": self.sharpness,
        }


def normalize_kernel_preset(value):
    """Return canonical lowercase kernelPreset name; raises ValueError if unknown."""
    if value is None:
        return DEFAULT_KERNEL_PRESET
    token = str(value).strip().lower()
    if not token:
        return DEFAULT_KERNEL_PRESET
    if token not in KERNEL_PRESETS:
        raise ValueError(
            f"kernelPreset must be one of {list(KERNEL_PRESETS)}, got {value!r}."
        )
    return token


def parse_kernel_preset(cfg):
    """Read and validate ``cfg['kernelPreset']`` from an init config dict."""
    if not isinstance(cfg, dict):
        return DEFAULT_KERNEL_PRESET
    return normalize_kernel_preset(cfg.get("kernelPreset", DEFAULT_KERNEL_PRESET))


def supports_botorch_rs(kernel_preset=DEFAULT_KERNEL_PRESET, parameter_gates=()):
    """True when ``kernel_preset`` and ``parameter_gates`` run on native Rust botorch_rs."""
    preset = normalize_kernel_preset(kernel_preset)
    if parameter_gates:
        return False
    return preset not in BOTORCH_RS_CUSTOM_PRESETS


def botorch_rs_unsupported_reason(kernel_preset=DEFAULT_KERNEL_PRESET, parameter_gates=()):
    """Reason string when settings require Python BoTorch/GPyTorch modules."""
    preset = normalize_kernel_preset(kernel_preset)
    if parameter_gates:
        return (
            "hierarchical parameter gating (gatedBy) uses a custom InputTransform "
            "not built into botorch_rs"
        )
    return (
        f"kernelPreset='{preset}' uses a custom shared-lengthscale additive GPyTorch kernel "
        "not built into botorch_rs"
    )


def _finite_float(raw, field_name, param_key, default):
    if raw is None or raw == "":
        return float(default)
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter '{param_key}' {field_name} must be a finite number, got {raw!r}."
        ) from exc
    if not math.isfinite(val):
        raise ValueError(
            f"Parameter '{param_key}' {field_name} must be finite, got {raw!r}."
        )
    return val


def parse_parameter_gates(parameters, parameter_names=None):
    """Parse and topologically order hierarchical parameter gates from ``init.parameters``.

    Validates that every ``gatedBy`` key exists (matched case-insensitively against
    ``parameter_names``), is not self-referential, has ``gateThreshold`` in [0, 1),
    ``gateNeutral`` in [0, 1], ``gateSharpness`` >= 0, and forms a directed acyclic graph (DAG).
    Returns a tuple of ``ParameterGate`` ordered parent-before-child so multi-level hierarchies
    cascade cleanly in a single pass.
    """
    parameters = list(parameters or [])
    if parameter_names is None:
        parameter_names = [str((p or {}).get("key") or "").strip() for p in parameters]
    else:
        parameter_names = [str(k).strip() for k in parameter_names]

    folded_to_index = {}
    for idx, name in enumerate(parameter_names):
        if name:
            folded_to_index[name.casefold()] = idx

    raw_gates = {}
    for idx, entry in enumerate(parameters):
        if not isinstance(entry, dict):
            continue
        child_key = parameter_names[idx] if idx < len(parameter_names) else str(entry.get("key") or "").strip()
        raw_parent = entry.get("gatedBy")
        parent_token = str(raw_parent).strip() if raw_parent is not None else ""
        if not parent_token:
            continue

        parent_idx = folded_to_index.get(parent_token.casefold())
        if parent_idx is None:
            raise ValueError(
                f"Parameter '{child_key}' gatedBy='{parent_token}' does not match any configured parameter."
            )
        if parent_idx == idx:
            raise ValueError(
                f"Parameter '{child_key}' cannot be gated by itself."
            )

        threshold = _finite_float(entry.get("gateThreshold"), "gateThreshold", child_key, 0.0)
        neutral = _finite_float(entry.get("gateNeutral"), "gateNeutral", child_key, 0.0)
        sharpness = _finite_float(entry.get("gateSharpness"), "gateSharpness", child_key, 0.0)

        if threshold < 0.0 or threshold >= 1.0:
            raise ValueError(
                f"Parameter '{child_key}' gateThreshold must lie in [0, 1), got {threshold}."
            )
        if neutral < 0.0 or neutral > 1.0:
            raise ValueError(
                f"Parameter '{child_key}' gateNeutral must lie in [0, 1], got {neutral}."
            )
        if sharpness < 0.0:
            raise ValueError(
                f"Parameter '{child_key}' gateSharpness must be >= 0, got {sharpness}."
            )

        raw_gates[idx] = ParameterGate(
            child_index=idx,
            child_key=child_key,
            parent_index=parent_idx,
            parent_key=parameter_names[parent_idx],
            threshold=threshold,
            neutral=neutral,
            sharpness=sharpness,
        )

    if not raw_gates:
        return ()

    # Detect cycles and order parent gates before child gates.
    state = {}  # idx -> 1 (visiting) | 2 (visited)
    ordered = []

    def visit(node_idx, trail):
        st = state.get(node_idx, 0)
        if st == 2:
            return
        if st == 1:
            cycle_start = trail.index(node_idx)
            cycle_names = [parameter_names[i] for i in trail[cycle_start:]] + [parameter_names[node_idx]]
            raise ValueError(
                "Parameter gating contains a cycle: " + " -> ".join(cycle_names) + "."
            )
        state[node_idx] = 1
        gate = raw_gates.get(node_idx)
        if gate is not None:
            if gate.parent_index in raw_gates:
                visit(gate.parent_index, trail + [node_idx])
            ordered.append(gate)
        state[node_idx] = 2

    for child_idx in sorted(raw_gates.keys()):
        visit(child_idx, [])

    return tuple(ordered)


def apply_parameter_gates_tensor(X, gates):
    """Apply ``gates`` out-of-place to normalized tensor ``X`` (..., d) or (..., d+1).

    Uses out-of-place column stacking so autograd works identically on both BoTorch and
    botorch-rust during L-BFGS-B acquisition optimization.
    """
    if not gates:
        return X
    import torch

    n_cols = int(X.shape[-1])
    cols = [X[..., j] for j in range(n_cols)]
    for gate in gates:
        if gate.parent_index >= n_cols or gate.child_index >= n_cols:
            continue
        u_parent = cols[gate.parent_index].clamp(0.0, 1.0)
        u_child = cols[gate.child_index]
        tau = gate.threshold
        if gate.sharpness > 0.0:
            k = gate.sharpness
            sig_0 = 1.0 / (1.0 + math.exp(k * tau))
            sig_1 = 1.0 / (1.0 + math.exp(-k * (1.0 - tau)))
            denom = max(sig_1 - sig_0, 1e-12)
            weight = ((torch.sigmoid(k * (u_parent - tau)) - sig_0) / denom).clamp(0.0, 1.0)
        else:
            denom = max(1.0 - tau, 1e-12)
            weight = ((u_parent - tau) / denom).clamp(0.0, 1.0)
        c0 = gate.neutral
        cols[gate.child_index] = c0 + weight * (u_child - c0)
    return torch.stack(cols, dim=-1)


_GATE_TRANSFORM_CLASSES = {}


def build_gate_input_transform(gates, library="botorch"):
    """Build a BoTorch / botorch_rs ``InputTransform`` for ``gates`` (or None when empty)."""
    gates = tuple(gates or ())
    if not gates:
        return None

    if library == "botorch_rs":
        from botorch_rs.models.transforms.input import InputTransform
    else:
        from botorch.models.transforms.input import InputTransform

    cls = _GATE_TRANSFORM_CLASSES.get(InputTransform)
    if cls is None:
        class _HierarchicalGateInputTransform(InputTransform):
            transform_on_train = True
            transform_on_eval = True
            transform_on_fantasize = True

            def __init__(self, parameter_gates):
                super().__init__()
                self.parameter_gates = tuple(parameter_gates)

            def transform(self, X):
                return apply_parameter_gates_tensor(X, self.parameter_gates)

        cls = _HierarchicalGateInputTransform
        _GATE_TRANSFORM_CLASSES[InputTransform] = cls

    return cls(gates)


_SHARED_ADD_KERNEL_CLASSES = {}


def _shared_additive_matern_kernel_class():
    """1-parameter 1D shared-lengthscale additive Matern kernel: (1/d) * sum_j k_nu(|x_j - x'_j| / l_1d)."""
    import torch
    from gpytorch.kernels import Kernel

    cls = _SHARED_ADD_KERNEL_CLASSES.get(Kernel)
    if cls is None:
        class _SharedLengthscaleAdditiveMaternKernel(Kernel):
            has_lengthscale = True

            def __init__(self, nu=1.5, **kwargs):
                super().__init__(**kwargs)
                self.nu = float(nu)

            def forward(self, x1, x2, diag=False, **params):
                if diag:
                    return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)
                ls = self.lengthscale.clamp_min(1e-5)
                diff = torch.abs(x1.unsqueeze(-2) - x2.unsqueeze(-3)) / ls.unsqueeze(-1)
                if self.nu == 0.5:
                    k_1d = torch.exp(-diff)
                elif self.nu == 1.5:
                    r = math.sqrt(3.0) * diff
                    k_1d = (1.0 + r) * torch.exp(-r)
                else:
                    r = math.sqrt(5.0) * diff
                    k_1d = (1.0 + r + (r ** 2) / 3.0) * torch.exp(-r)
                return torch.mean(k_1d, dim=-1)

        cls = _SharedLengthscaleAdditiveMaternKernel
        _SHARED_ADD_KERNEL_CLASSES[Kernel] = cls
    return cls


def _aug_batch_shape(train_x, train_y):
    import torch

    input_batch = train_x.shape[:-2]
    num_outputs = int(train_y.shape[-1])
    if num_outputs > 1:
        return torch.Size(tuple(input_batch) + (num_outputs,))
    return torch.Size(tuple(input_batch))


def build_covar_and_likelihood(preset, d, batch_shape, library="botorch"):
    """Construct ``(covar_module, likelihood)`` for ``preset`` on ``library`` ('botorch' or 'botorch_rs').

    Returns ``(None, None)`` when ``preset == 'default'`` so ``SingleTaskGP`` uses its stock defaults.
    """
    preset = normalize_kernel_preset(preset)
    if preset == DEFAULT_KERNEL_PRESET:
        return None, None

    if library == "botorch_rs":
        if not supports_botorch_rs(preset):
            raise ValueError(botorch_rs_unsupported_reason(preset))
        from botorch_rs.constraints import GreaterThan
        from botorch_rs.kernels import MaternKernel, RQKernel, ScaleKernel
        from botorch_rs.likelihoods import GaussianLikelihood
        from botorch_rs.priors import GammaPrior, LogNormalPrior
    else:
        from gpytorch.constraints import GreaterThan
        from gpytorch.kernels import MaternKernel, RQKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.priors import GammaPrior, LogNormalPrior

    d = max(1, int(d))
    mu_d = math.log(math.sqrt(2.0)) + 0.5 * math.log(d)

    if preset in ("matern52_ard", "matern32_ard", "matern12_ard"):
        nu = {"matern52_ard": 2.5, "matern32_ard": 1.5, "matern12_ard": 0.5}[preset]
        covar = ScaleKernel(
            MaternKernel(
                nu=nu,
                ard_num_dims=d,
                batch_shape=batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(1.1, 0.05),
            batch_shape=batch_shape,
            noise_constraint=GreaterThan(1e-4),
        )
        return covar, likelihood

    # Noisy HITL / experimental likelihood prior (median noise variance ~0.30 of standardized Y).
    likelihood = GaussianLikelihood(
        noise_prior=LogNormalPrior(-1.2, 0.9),
        batch_shape=batch_shape,
        noise_constraint=GreaterThan(1e-3),
    )
    ls_prior_d = LogNormalPrior(mu_d, 0.6)

    if preset == "matern32_dimscaled_ard":
        covar = ScaleKernel(
            MaternKernel(
                nu=1.5,
                ard_num_dims=d,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior_d,
                lengthscale_constraint=GreaterThan(0.025),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 2.0),
        )
        return covar, likelihood

    if preset == "matern32_iso":
        covar = ScaleKernel(
            MaternKernel(
                nu=1.5,
                ard_num_dims=None,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior_d,
                lengthscale_constraint=GreaterThan(0.025),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 2.0),
        )
        return covar, likelihood

    if preset == "rq_iso":
        covar = ScaleKernel(
            RQKernel(
                ard_num_dims=None,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior_d,
                lengthscale_constraint=GreaterThan(0.025),
                alpha_constraint=GreaterThan(0.05),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 2.0),
        )
        return covar, likelihood

    shared_cls = _shared_additive_matern_kernel_class()
    ls_prior_1d = LogNormalPrior(-0.2, 0.6)

    if preset == "shared_add_matern32":
        covar = ScaleKernel(
            shared_cls(
                nu=1.5,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior_1d,
                lengthscale_constraint=GreaterThan(0.025),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 2.0),
        )
        return covar, likelihood

    if preset == "shared_add_inter_matern32":
        k_add = ScaleKernel(
            shared_cls(
                nu=1.5,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior_1d,
                lengthscale_constraint=GreaterThan(0.025),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(1.5, 2.0),
        )
        k_inter = ScaleKernel(
            MaternKernel(
                nu=1.5,
                ard_num_dims=None,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior_d,
                lengthscale_constraint=GreaterThan(0.025),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(1.5, 2.0),
        )
        return k_add + k_inter, likelihood

    raise ValueError(f"Unhandled kernelPreset {preset!r}")


def build_single_task_gp(train_x, train_obj, *, library="botorch", kernel_preset=DEFAULT_KERNEL_PRESET,
                         parameter_gates=()):
    """Construct ``(mll, model)`` for plain (non-contextual) BoTorch / botorch-rust runs.

    When ``kernel_preset == 'default'`` and ``parameter_gates`` is empty, calls
    ``SingleTaskGP(train_x, train_obj)`` with no optional keyword arguments so existing runs
    and lightweight test stubs behave identically bit-for-bit.
    """
    import bo_library

    preset = normalize_kernel_preset(kernel_preset)
    gates = tuple(parameter_gates or ())
    names = bo_library.namespace(library)

    if preset == DEFAULT_KERNEL_PRESET and not gates:
        model = names.SingleTaskGP(train_x, train_obj)
        mll = names.ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    d = int(train_x.shape[-1])
    batch_shape = _aug_batch_shape(train_x, train_obj)
    covar, likelihood = build_covar_and_likelihood(preset, d, batch_shape, library=library)
    input_transform = build_gate_input_transform(gates, library=library)

    kwargs = {}
    if covar is not None:
        kwargs["covar_module"] = covar
    if likelihood is not None:
        kwargs["likelihood"] = likelihood
    if input_transform is not None:
        kwargs["input_transform"] = input_transform

    model = names.SingleTaskGP(train_x, train_obj, **kwargs)
    mll = names.ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model
