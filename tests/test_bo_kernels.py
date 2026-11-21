"""Unit and real-stack tests for Backend/bo_kernels.py and its integration with bo_library."""

import math
import os
import pathlib
import sys
import unittest
from unittest import mock

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from _stubs import FakeTensor, install_stub_modules  # noqa: E402

_REAL_MODULE_ROOTS = (
    "torch", "botorch", "gpytorch", "linear_operator", "moocore", "pandas", "botorch_rs", "openbo"
)

try:
    import torch  # noqa: F401
    import botorch  # noqa: F401
    import botorch.fit  # noqa: F401
    import botorch.models  # noqa: F401
    import botorch.models.transforms.input  # noqa: F401
    import gpytorch.constraints  # noqa: F401
    import gpytorch.kernels  # noqa: F401
    import gpytorch.likelihoods  # noqa: F401
    import gpytorch.mlls  # noqa: F401
    import gpytorch.priors  # noqa: F401

    HAS_REAL_STACK = True
except ImportError:
    HAS_REAL_STACK = False

HAS_BOTORCH_RS = False
if HAS_REAL_STACK:
    try:
        import botorch_rs  # noqa: F401
        import botorch_rs.fit  # noqa: F401
        import botorch_rs.kernels  # noqa: F401
        import botorch_rs.mlls  # noqa: F401
        import botorch_rs.models  # noqa: F401

        HAS_BOTORCH_RS = True
    except ImportError:
        pass

_REAL_MODULES = {
    name: mod
    for name, mod in sys.modules.items()
    if HAS_REAL_STACK and name.split(".")[0] in _REAL_MODULE_ROOTS and mod is not None
}


def _restore_real_modules():
    sys.modules.update(_REAL_MODULES)
    for name in list(sys.modules):
        if name.split(".")[0] not in _REAL_MODULE_ROOTS or name in _REAL_MODULES:
            continue
        if getattr(sys.modules[name], "__spec__", None) is None:
            del sys.modules[name]


import bo_kernels  # noqa: E402
import bo_library  # noqa: E402


class KernelPresetParsingTests(unittest.TestCase):
    def test_accepts_all_canonical_presets_and_defaults_empty_tokens(self):
        self.assertEqual(bo_kernels.normalize_kernel_preset(None), "default")
        self.assertEqual(bo_kernels.normalize_kernel_preset(""), "default")
        self.assertEqual(bo_kernels.normalize_kernel_preset("   "), "default")
        for preset in bo_kernels.KERNEL_PRESETS:
            self.assertEqual(bo_kernels.normalize_kernel_preset(f"  {preset.upper()} "), preset)
            self.assertEqual(bo_kernels.parse_kernel_preset({"kernelPreset": preset}), preset)

    def test_rejects_unknown_preset(self):
        with self.assertRaisesRegex(ValueError, "kernelPreset must be one of"):
            bo_kernels.normalize_kernel_preset("spectral_mixture")


class ParameterGateParsingTests(unittest.TestCase):
    def test_orders_cascading_gates_parent_before_child(self):
        # Even though 'size' appears before 'alpha', 'visible -> alpha' must run before 'alpha -> size'.
        params = [
            {"key": "size", "init": {"low": 0, "high": 10}, "gatedBy": "ALPHA", "gateThreshold": 0.05},
            {"key": "alpha", "init": {"low": 0, "high": 1}, "gatedBy": "visible", "gateThreshold": 0.5, "gateSharpness": 20.0},
            {"key": "visible", "init": {"low": 0, "high": 1}},
            {"key": "hue", "init": {"low": 0, "high": 360}, "gatedBy": "alpha", "gateNeutral": 0.5},
        ]
        gates = bo_kernels.parse_parameter_gates(params)
        self.assertEqual([g.child_key for g in gates[:2]], ["alpha", "size"])
        self.assertEqual(gates[0].parent_key, "visible")
        self.assertEqual(gates[1].parent_key, "alpha")
        self.assertEqual(gates[2].neutral, 0.5)

    def test_rejects_unknown_self_cyclic_and_out_of_range_gates(self):
        bad_cases = [
            ([{"key": "x", "gatedBy": "missing"}], "does not match"),
            ([{"key": "x", "gatedBy": "x"}], "itself"),
            ([{"key": "a", "gatedBy": "b"}, {"key": "b", "gatedBy": "a"}], "cycle"),
            (
                [{"key": "a", "gatedBy": "c"}, {"key": "b", "gatedBy": "a"}, {"key": "c", "gatedBy": "b"}],
                "cycle",
            ),
            ([{"key": "a"}, {"key": "b", "gatedBy": "a", "gateThreshold": 1.0}], "gateThreshold"),
            ([{"key": "a"}, {"key": "b", "gatedBy": "a", "gateThreshold": -0.1}], "gateThreshold"),
            ([{"key": "a"}, {"key": "b", "gatedBy": "a", "gateNeutral": 1.2}], "gateNeutral"),
            ([{"key": "a"}, {"key": "b", "gatedBy": "a", "gateSharpness": -1.0}], "gateSharpness"),
            ([{"key": "a"}, {"key": "b", "gatedBy": "a", "gateNeutral": math.nan}], "finite"),
        ]
        for params, pattern in bad_cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                bo_kernels.parse_parameter_gates(params)


class LibraryResolutionWithKernelsTests(unittest.TestCase):
    def test_custom_additive_and_gated_runs_fall_back_to_botorch_cleanly(self):
        gates = bo_kernels.parse_parameter_gates([{"key": "a"}, {"key": "b", "gatedBy": "a"}])
        with mock.patch.object(bo_library, "botorch_rs_status", return_value=(True, "0.3.0", "")):
            lib, _ = bo_library.resolve("botorch_rs", kernel_preset="matern32_dimscaled_ard")
            self.assertEqual(lib, "botorch_rs")

            lib_add, reason_add = bo_library.resolve("botorch_rs", kernel_preset="shared_add_inter_matern32")
            self.assertEqual(lib_add, "botorch")
            self.assertIn("shared_add_inter_matern32", reason_add)

            lib_gate, reason_gate = bo_library.resolve("auto", kernel_preset="matern32_iso", parameter_gates=gates)
            self.assertEqual(lib_gate, "botorch")
            self.assertIn("gatedBy", reason_gate)


class StubModelConstructionTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(sys.modules)
        install_stub_modules()

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._saved)
        if HAS_REAL_STACK:
            _restore_real_modules()

    def test_default_preset_leaves_stock_singletaskgp_untouched_and_presets_attach_modules(self):
        x = FakeTensor([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]])
        y = FakeTensor([[0.2], [0.9]])

        _, default_model = bo_kernels.build_single_task_gp(x, y, library="botorch", kernel_preset="default")
        self.assertIsNone(default_model.covar_module)
        self.assertIsNone(default_model.input_transform)

        gates = bo_kernels.parse_parameter_gates([{"key": "p0"}, {"key": "p1", "gatedBy": "p0"}, {"key": "p2"}])
        for preset in bo_kernels.KERNEL_PRESETS:
            if preset == "default":
                continue
            with self.subTest(preset=preset):
                _, model = bo_kernels.build_single_task_gp(
                    x, y, library="botorch", kernel_preset=preset, parameter_gates=gates
                )
                self.assertIsNotNone(model.covar_module)
                self.assertIsNotNone(model.input_transform)


@unittest.skipUnless(HAS_REAL_STACK, "Real torch/botorch/gpytorch stack not installed")
class RealStackKernelAndGatingTests(unittest.TestCase):
    def setUp(self):
        _restore_real_modules()

    def test_hierarchical_gate_anchors_cascade_and_preserve_gradients(self):
        import torch

        params = [
            {"key": "visible"},
            {"key": "alpha", "gatedBy": "visible", "gateThreshold": 0.5, "gateNeutral": 0.0, "gateSharpness": 20.0},
            {"key": "size", "gatedBy": "alpha", "gateThreshold": 0.0, "gateNeutral": 0.0},
            {"key": "saturation"},
            {"key": "hue", "gatedBy": "saturation", "gateThreshold": 0.0, "gateNeutral": 0.5},
        ]
        gates = bo_kernels.parse_parameter_gates(params)
        x = torch.tensor(
            [
                [0.0, 0.9, 0.8, 0.0, 0.1],  # visible=0 -> alpha->0 -> size->0; sat=0 -> hue->0.5
                [1.0, 0.5, 0.8, 1.0, 0.1],  # visible=1 -> alpha=0.5 -> size=0.4; sat=1 -> hue=0.1
            ],
            dtype=torch.double,
            requires_grad=True,
        )
        warped = bo_kernels.apply_parameter_gates_tensor(x, gates)
        self.assertAlmostEqual(float(warped[0, 1]), 0.0, places=9)
        self.assertAlmostEqual(float(warped[0, 2]), 0.0, places=9)
        self.assertAlmostEqual(float(warped[0, 4]), 0.5, places=9)
        self.assertAlmostEqual(float(warped[1, 1]), 0.5, places=9)
        self.assertAlmostEqual(float(warped[1, 2]), 0.4, places=9)
        self.assertAlmostEqual(float(warped[1, 4]), 0.1, places=9)

        warped.sum().backward()
        self.assertTrue(torch.all(torch.isfinite(x.grad)))

    def test_all_presets_fit_single_and_multi_objective_gps_on_botorch(self):
        import torch

        torch.manual_seed(7)
        x = torch.rand(10, 4, dtype=torch.double)
        y_so = torch.randn(10, 1, dtype=torch.double)
        y_mo = torch.randn(10, 2, dtype=torch.double)
        gates = bo_kernels.parse_parameter_gates(
            [{"key": "p0"}, {"key": "p1", "gatedBy": "p0", "gateNeutral": 0.5}, {"key": "p2"}, {"key": "p3"}]
        )
        names = bo_library.namespace("botorch")

        for preset in bo_kernels.KERNEL_PRESETS:
            for y in (y_so, y_mo):
                with self.subTest(preset=preset, m=int(y.shape[-1])):
                    mll, model = bo_kernels.build_single_task_gp(
                        x, y, library="botorch", kernel_preset=preset, parameter_gates=gates
                    )
                    names.fit_gpytorch_mll(mll)
                    post = model.posterior(x[:3])
                    self.assertEqual(tuple(post.mean.shape), (3, int(y.shape[-1])))
                    self.assertTrue(torch.all(torch.isfinite(post.mean)))

    @unittest.skipUnless(HAS_BOTORCH_RS, "botorch_rs not installed")
    def test_supported_presets_fit_on_botorch_rs(self):
        import torch

        torch.manual_seed(11)
        x = torch.rand(10, 3, dtype=torch.double)
        y = torch.randn(10, 2, dtype=torch.double)
        names = bo_library.namespace("botorch_rs")
        for preset in bo_kernels.KERNEL_PRESETS:
            if not bo_kernels.supports_botorch_rs(preset):
                continue
            with self.subTest(preset=preset):
                mll, model = bo_kernels.build_single_task_gp(x, y, library="botorch_rs", kernel_preset=preset)
                names.fit_gpytorch_mll(mll)
                post = model.posterior(x[:2])
                self.assertEqual(tuple(post.mean.shape), (2, 2))
                self.assertTrue(torch.all(torch.isfinite(post.mean)))


if __name__ == "__main__":
    unittest.main()
