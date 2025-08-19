# -*- coding: utf-8 -*-
"""
RES4LYF loop helpers for ComfyUI (beta-aligned, with combo & sampler_mode loop).

Adds:
1) RES4LYFComboLoop   -> loop sampler + scheduler in one node
2) Sampler mode loop  -> toggleable on every node (sequential/random/ping_pong, with skip)
3) Fallback enums     -> used if upstream imports fail
4) Correct total_combinations formulas
5) NEW: label output on SchedulerLoop and SamplerLoop

Exact enum function names (used in RETURN_TYPES):
- get_sampler_name_list()
- get_res4lyf_scheduler_list()
"""

from __future__ import annotations

import sys
import types
import random
import threading
import time
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from typing import List

# ---------------------------------------------------------------------
# Paths (fixed to your layout)
# ---------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_RES4LYF_ROOT = _THIS_DIR.parent / "RES4LYF"
_BETA_DIR = _RES4LYF_ROOT / "beta"
_HELPER_FILE = _RES4LYF_ROOT / "helper.py"
_RK_FILE = _BETA_DIR / "rk_coefficients_beta.py"

# ---------------------------------------------------------------------
# Fallback enums (used only if upstream import/derivation fails)
# Keep these in sync with ClownsharKSampler_Beta expectations.
# ---------------------------------------------------------------------

SAMPLER_MODE_ENUM = ["unsample", "standard", "resample"]

FALLBACK_SAMPLERS = [
    "none",
    # multistep
    "multistep/res_2m", "multistep/res_3m", "multistep/dpmpp_2m", "multistep/dpmpp_3m",
    "multistep/abnorsett_2m", "multistep/abnorsett_3m", "multistep/abnorsett_4m",
    "multistep/deis_2m", "multistep/deis_3m", "multistep/deis_4m",
    # exponential
    "exponential/res_2s_rkmk2e", "exponential/res_2s", "exponential/res_2s_stable",
    "exponential/res_3s", "exponential/res_3s_non-monotonic", "exponential/res_3s_alt",
    "exponential/res_3s_cox_matthews", "exponential/res_3s_lie", "exponential/res_3s_sunstar",
    "exponential/res_3s_strehmel_weiner",
    "exponential/res_4s_krogstad", "exponential/res_4s_krogstad_alt",
    "exponential/res_4s_strehmel_weiner", "exponential/res_4s_strehmel_weiner_alt",
    "exponential/res_4s_cox_matthews", "exponential/res_4s_cfree4", "exponential/res_4s_friedli",
    "exponential/res_4s_minchev", "exponential/res_4s_munthe-kaas",
    "exponential/res_5s", "exponential/res_5s_hochbruck-ostermann",
    "exponential/res_6s", "exponential/res_8s", "exponential/res_8s_alt",
    "exponential/res_10s", "exponential/res_15s", "exponential/res_16s",
    "exponential/etdrk2_2s", "exponential/etdrk3_a_3s", "exponential/etdrk3_b_3s",
    "exponential/etdrk4_4s", "exponential/etdrk4_4s_alt",
    "exponential/dpmpp_2s", "exponential/dpmpp_sde_2s", "exponential/dpmpp_3s",
    "exponential/lawson2a_2s", "exponential/lawson2b_2s", "exponential/lawson4_4s",
    "exponential/lawson41-gen_4s", "exponential/lawson41-gen-mod_4s",
    "exponential/ddim",
    # hybrid
    "hybrid/pec423_2h2s", "hybrid/pec433_2h3s",
    "hybrid/abnorsett2_1h2s", "hybrid/abnorsett3_2h2s", "hybrid/abnorsett4_3h2s",
    "hybrid/lawson42-gen-mod_1h4s", "hybrid/lawson43-gen-mod_2h4s",
    "hybrid/lawson44-gen-mod_3h4s", "hybrid/lawson45-gen-mod_4h4s",
    # linear (explicit RK)
    "linear/ralston_2s", "linear/ralston_3s", "linear/ralston_4s",
    "linear/midpoint_2s", "linear/heun_2s", "linear/heun_3s", "linear/houwen-wray_3s",
    "linear/kutta_3s", "linear/ssprk3_3s", "linear/ssprk4_4s",
    "linear/rk38_4s", "linear/rk4_4s", "linear/rk5_7s", "linear/rk6_7s",
    "linear/bogacki-shampine_4s", "linear/bogacki-shampine_7s",
    "linear/dormand-prince_6s", "linear/dormand-prince_13s", "linear/tsi_7s", "linear/euler",
    # diag_implicit
    "diag_implicit/irk_exp_diag_2s", "diag_implicit/kraaijevanger_spijker_2s",
    "diag_implicit/qin_zhang_2s", "diag_implicit/pareschi_russo_2s",
    "diag_implicit/pareschi_russo_alt_2s", "diag_implicit/crouzeix_2s",
    "diag_implicit/crouzeix_3s", "diag_implicit/crouzeix_3s_alt",
    # fully_implicit
    "fully_implicit/gauss-legendre_2s", "fully_implicit/gauss-legendre_3s",
    "fully_implicit/gauss-legendre_4s", "fully_implicit/gauss-legendre_4s_alternating_a",
    "fully_implicit/gauss-legendre_4s_ascending_a", "fully_implicit/gauss-legendre_4s_alt",
    "fully_implicit/gauss-legendre_5s", "fully_implicit/gauss-legendre_5s_ascending",
    "fully_implicit/radau_ia_2s", "fully_implicit/radau_ia_3s",
    "fully_implicit/radau_iia_2s", "fully_implicit/radau_iia_3s", "fully_implicit/radau_iia_3s_alt",
    "fully_implicit/radau_iia_5s", "fully_implicit/radau_iia_7s", "fully_implicit/radau_iia_9s",
    "fully_implicit/radau_iia_11s",
    "fully_implicit/lobatto_iiia_2s", "fully_implicit/lobatto_iiia_3s", "fully_implicit/lobatto_iiia_4s",
    "fully_implicit/lobatto_iiib_2s", "fully_implicit/lobatto_iiib_3s", "fully_implicit/lobatto_iiib_4s",
    "fully_implicit/lobatto_iiic_2s", "fully_implicit/lobatto_iiic_3s", "fully_implicit/lobatto_iiic_4s",
    "fully_implicit/lobatto_iiic_star_2s", "fully_implicit/lobatto_iiic_star_3s",
    "fully_implicit/lobatto_iiid_2s", "fully_implicit/lobatto_iiid_3s",
]

FALLBACK_SCHEDULERS = [
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta",
    "normal", "linear_quadratic", "kl_optimal", "bong_tangent", "beta57"
]

# ---------------------------------------------------------------------
# Inject package so beta/ relative imports work, then import modules
# Use graceful fallbacks if anything fails.
# ---------------------------------------------------------------------

_rk_mod = None
_helper_mod = None

try:
    if _RK_FILE.is_file() and _HELPER_FILE.is_file():
        if "RES4LYF" not in sys.modules:
            pkg = types.ModuleType("RES4LYF")
            pkg.__path__ = [str(_RES4LYF_ROOT)]
            sys.modules["RES4LYF"] = pkg
        else:
            mod = sys.modules["RES4LYF"]
            if not hasattr(mod, "__path__"):
                mod.__path__ = [str(_RES4LYF_ROOT)]
            elif str(_RES4LYF_ROOT) not in mod.__path__:      # type: ignore[attr-defined]
                mod.__path__.append(str(_RES4LYF_ROOT))       # type: ignore[attr-defined]

        if "RES4LYF.beta" not in sys.modules:
            beta_pkg = types.ModuleType("RES4LYF.beta")
            beta_pkg.__path__ = [str(_BETA_DIR)]
            sys.modules["RES4LYF.beta"] = beta_pkg
        else:
            mod = sys.modules["RES4LYF.beta"]
            if not hasattr(mod, "__path__"):
                mod.__path__ = [str(_BETA_DIR)]
            elif str(_BETA_DIR) not in mod.__path__:          # type: ignore[attr-defined]
                mod.__path__.append(str(_BETA_DIR))           # type: ignore[attr-defined]

        def _import_as(dotted_name: str, file_path: Path):
            spec = spec_from_file_location(dotted_name, str(file_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed spec for {file_path}")
            module = module_from_spec(spec)
            sys.modules[dotted_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module

        _rk_mod = _import_as("RES4LYF.beta.rk_coefficients_beta", _RK_FILE)
        _helper_mod = _import_as("RES4LYF.helper", _HELPER_FILE)
except Exception:
    _rk_mod = None
    _helper_mod = None

# ---------------------------------------------------------------------
# Build enums (prefer upstream, else fallback)
# ---------------------------------------------------------------------

def _derive_sampler_enum() -> List[str]:
    if _rk_mod and hasattr(_rk_mod, "RK_SAMPLER_NAMES_BETA_FOLDERS"):
        names = list(getattr(_rk_mod, "RK_SAMPLER_NAMES_BETA_FOLDERS"))
        out = ["none"]
        seen = set(out)
        for n in names:
            s = n[1:] if isinstance(n, str) and n.startswith("/") else n
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    if _rk_mod and hasattr(_rk_mod, "get_sampler_name_list"):
        try:
            raw = list(_rk_mod.get_sampler_name_list())
            if raw and raw[0] != "none":
                raw = ["none"] + [r[1:] if isinstance(r, str) and r.startswith("/") else r for r in raw]
            return raw
        except Exception:
            pass
    return FALLBACK_SAMPLERS

def _derive_scheduler_enum() -> List[str]:
    if _helper_mod:
        for fname in (
            "get_res4lyf_scheduler_list",
            "get_scheduler_list",
            "get_scheduler_name_list",
            "get_res4lyf_scheduler_names",
        ):
            fn = getattr(_helper_mod, fname, None)
            if callable(fn):
                try:
                    return list(fn())
                except Exception:
                    pass
    return FALLBACK_SCHEDULERS

_SAMPLER_ENUM = _derive_sampler_enum()
_SCHEDULER_ENUM = _derive_scheduler_enum()

def _gen_int_range(start: int, end: int, step: int) -> list:
    """Inclusive integer range that supports ascending or descending."""
    if step == 0:
        step = 1
    sign = 1 if end >= start else -1
    step = abs(step) * sign
    stop = end + sign  # inclusive
    return list(range(int(start), int(stop), int(step)))

def _gen_float_range(start: float, end: float, step: float, decimals: int = 6) -> list:
    """Inclusive float range (stable), supports ascending/descending."""
    eps = 10**(-decimals) / 2.0
    if step == 0:
        step = 1.0
    # normalize direction
    if end >= start and step < 0:
        step = abs(step)
    if end < start and step > 0:
        step = -abs(step)

    vals = []
    cur = float(start)
    # use direction-aware comparison
    if step > 0:
        while cur <= end + eps:
            r = round(cur, decimals)
            # avoid -0.0
            if r == 0:
                r = 0.0
            vals.append(r)
            cur = round(cur + step, decimals)
    else:
        while cur >= end - eps:
            r = round(cur, decimals)
            if r == 0:
                r = 0.0
            vals.append(r)
            cur = round(cur + step, decimals)
    return vals

# ---------------------------------------------------------------------
# Export EXACT names used in RETURN_TYPES
# ---------------------------------------------------------------------

def get_sampler_name_list():
    return _SAMPLER_ENUM

def get_res4lyf_scheduler_list():
    return _SCHEDULER_ENUM

# ---------------------------------------------------------------------
# Common utils
# ---------------------------------------------------------------------

def _parse_skip_list(text: str) -> List[str]:
    if not text:
        return []
    seps = [",", "\n", "\r", "\t", " "]
    parts = [text]
    for s in seps:
        nxt = []
        for p in parts:
            nxt.extend(p.split(s))
        parts = nxt
    return [t.strip() for t in parts if t.strip()]

def _advance_counter(group_state, mode: str):
    exec_id = f"{time.time()}_{threading.current_thread().ident}_{mode}"
    if group_state["last_exec"].get(mode) != exec_id:
        group_state["last_exec"][mode] = exec_id
        if mode in group_state["first_flag"]:
            group_state["counters"][mode] += 1
        else:
            group_state["first_flag"].add(mode)

def _select_index(total: int, step: int, mode: str, rng: random.Random) -> int:
    if total <= 0:
        return 0
    if mode == "random":
        return rng.randrange(total)
    if mode == "ping_pong":
        period = max(1, total * 2 - 2)
        t = step % period
        return t if t < total else period - t
    return step % total  # sequential

def _available_modes(skip_modes_text: str) -> List[str]:
    skip = set(_parse_skip_list(skip_modes_text))
    modes = [m for m in SAMPLER_MODE_ENUM if m not in skip]
    return modes or list(SAMPLER_MODE_ENUM)

def _loop_sampler_mode(loop_enabled: bool, loop_method: str, seed: int,
                       skip_modes_text: str, group_state):
    modes_all = _available_modes(skip_modes_text)
    rng = random.Random(seed)
    if not loop_enabled:
        return "standard", 0, len(modes_all)  # not looping: treat multiplier as 1 by caller
    _advance_counter(group_state, loop_method)
    step = group_state["counters"][loop_method]
    idx = _select_index(len(modes_all), step, loop_method, rng)
    return modes_all[idx], idx, len(modes_all)

# ---------------------------------------------------------------------
# Scheduler Loop  (now with label)
# ---------------------------------------------------------------------

class RES4LYFSchedulerLoop:
    """
    Loop through schedulers only.
    Output #1 'scheduler' is an ENUM from get_res4lyf_scheduler_list().
    Also outputs 'sampler_mode' (toggleable looping) and 'label'.
    """
    _sched_state = {"counters": {"sequential": 0, "random": 0, "ping_pong": 0},
                    "last_exec": {"sequential": None, "random": None, "ping_pong": None},
                    "first_flag": set()}
    _mode_state  = {"counters": {"sequential": 0, "random": 0, "ping_pong": 0},
                    "last_exec": {"sequential": None, "random": None, "ping_pong": None},
                    "first_flag": set()}

    RETURN_TYPES = (get_res4lyf_scheduler_list(), "INT", "INT", "STRING", SAMPLER_MODE_ENUM, "STRING")
    RETURN_NAMES  = ("scheduler", "current_index", "total_combinations", "current_combination", "sampler_mode", "label")
    FUNCTION = "loop_res4lyf_scheduler"
    CATEGORY = "RES4LYF/Loop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["sequential", "random", "ping_pong"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "skip_schedulers": ("STRING", {"default": "", "multiline": True}),
                "loop_sampler_mode": ("BOOLEAN", {"default": False}),
                "sampler_mode_method": (["sequential", "random", "ping_pong"],),
                "skip_sampler_modes": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def loop_res4lyf_scheduler(self, mode, seed, reset=False, skip_schedulers="",
                               loop_sampler_mode=False, sampler_mode_method="sequential",
                               skip_sampler_modes=""):
        rng = random.Random(seed)

        sched_all_raw = list(get_res4lyf_scheduler_list())
        skip = set(_parse_skip_list(skip_schedulers))
        sched_all = [s for s in sched_all_raw if s not in skip] or sched_all_raw

        if reset:
            RES4LYFSchedulerLoop._sched_state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            RES4LYFSchedulerLoop._sched_state["first_flag"].clear()

        _advance_counter(RES4LYFSchedulerLoop._sched_state, mode)
        step = RES4LYFSchedulerLoop._sched_state["counters"][mode]
        idx = _select_index(len(sched_all), step, mode, rng)
        scheduler = sched_all[idx]

        # sampler_mode looping
        if reset:
            RES4LYFSchedulerLoop._mode_state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            RES4LYFSchedulerLoop._mode_state["first_flag"].clear()
        sampler_mode, _, modes_total = _loop_sampler_mode(loop_sampler_mode, sampler_mode_method, seed, skip_sampler_modes, RES4LYFSchedulerLoop._mode_state)

        total_combinations = len(sched_all) * (modes_total if loop_sampler_mode else 1)
        label = f"{scheduler} | {sampler_mode}"

        return (scheduler, idx, total_combinations, f"{scheduler}", sampler_mode, label)

# ---------------------------------------------------------------------
# Sampler Loop  (now with label)
# ---------------------------------------------------------------------

class RES4LYFSamplerLoop:
    """
    Loop through samplers only (Beta list).
    Output #1 'sampler_name' is an ENUM from get_sampler_name_list().
    Also outputs 'sampler_mode' (toggleable looping) and 'label'.
    """
    _samp_state = {"counters": {"sequential": 0, "random": 0, "ping_pong": 0},
                   "last_exec": {"sequential": None, "random": None, "ping_pong": None},
                   "first_flag": set()}
    _mode_state  = {"counters": {"sequential": 0, "random": 0, "ping_pong": 0},
                    "last_exec": {"sequential": None, "random": None, "ping_pong": None},
                    "first_flag": set()}

    RETURN_TYPES = (get_sampler_name_list(), "INT", "INT", "STRING", SAMPLER_MODE_ENUM, "STRING")
    RETURN_NAMES  = ("sampler_name", "current_index", "total_combinations", "current_combination", "sampler_mode", "label")
    FUNCTION = "loop_res4lyf_sampler"
    CATEGORY = "RES4LYF/Loop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["sequential", "random", "ping_pong"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "skip_samplers": ("STRING", {"default": "", "multiline": True}),
                "loop_sampler_mode": ("BOOLEAN", {"default": False}),
                "sampler_mode_method": (["sequential", "random", "ping_pong"],),
                "skip_sampler_modes": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def loop_res4lyf_sampler(self, mode, seed, reset=False, skip_samplers="",
                             loop_sampler_mode=False, sampler_mode_method="sequential",
                             skip_sampler_modes=""):
        rng = random.Random(seed)

        samp_all_raw = list(get_sampler_name_list())
        skip = set(_parse_skip_list(skip_samplers))
        samp_all = [s for s in samp_all_raw if s not in skip] or samp_all_raw

        if reset:
            RES4LYFSamplerLoop._samp_state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            RES4LYFSamplerLoop._samp_state["first_flag"].clear()

        _advance_counter(RES4LYFSamplerLoop._samp_state, mode)
        step = RES4LYFSamplerLoop._samp_state["counters"][mode]
        idx = _select_index(len(samp_all), step, mode, rng)
        sampler = samp_all[idx]

        # sampler_mode looping
        if reset:
            RES4LYFSamplerLoop._mode_state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            RES4LYFSamplerLoop._mode_state["first_flag"].clear()
        sampler_mode, _, modes_total = _loop_sampler_mode(loop_sampler_mode, sampler_mode_method, seed, skip_sampler_modes, RES4LYFSamplerLoop._mode_state)

        total_combinations = len(samp_all) * (modes_total if loop_sampler_mode else 1)
        label = f"{sampler} | {sampler_mode}"

        return (sampler, idx, total_combinations, f"{sampler}", sampler_mode, label)

# ---------------------------------------------------------------------
# Combo Loop (sampler + scheduler + sampler_mode)  — unchanged
# ---------------------------------------------------------------------

class RES4LYFComboLoop:
    """
    Loop sampler and scheduler (and sampler_mode) in one node.
    Outputs:
      1: sampler_name   (enum get_sampler_name_list())
      2: scheduler      (enum get_res4lyf_scheduler_list())
      3: sampler_mode   (enum SAMPLER_MODE_ENUM)
      4: sampler_index
      5: scheduler_index
      6: sampler_mode_index
      7: total_combinations
      8: label
    """

    _samp_state = {"counters": {"sequential": 0, "random": 0, "ping_pong": 0},
                   "last_exec": {"sequential": None, "random": None, "ping_pong": None},
                   "first_flag": set()}
    _sched_state = {"counters": {"sequential": 0, "random": 0, "ping_pong": 0},
                    "last_exec": {"sequential": None, "random": None, "ping_pong": None},
                    "first_flag": set()}
    _mode_state  = {"counters": {"sequential": 0, "random": 0, "ping_pong": 0},
                    "last_exec": {"sequential": None, "random": None, "ping_pong": None},
                    "first_flag": set()}

    RETURN_TYPES = (get_sampler_name_list(), get_res4lyf_scheduler_list(), SAMPLER_MODE_ENUM, "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("sampler_name", "scheduler", "sampler_mode", "sampler_index", "scheduler_index", "sampler_mode_index", "total_combinations", "label")
    FUNCTION = "loop_res4lyf_combo"
    CATEGORY = "RES4LYF/Loop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_loop_mode": (["sequential", "random", "ping_pong"],),
                "scheduler_loop_mode": (["sequential", "random", "ping_pong"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "skip_samplers": ("STRING", {"default": "", "multiline": True}),
                "skip_schedulers": ("STRING", {"default": "", "multiline": True}),
                "loop_sampler_mode": ("BOOLEAN", {"default": False}),
                "sampler_mode_method": (["sequential", "random", "ping_pong"],),
                "skip_sampler_modes": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def loop_res4lyf_combo(self, sampler_loop_mode, scheduler_loop_mode, seed, reset=False,
                           skip_samplers="", skip_schedulers="",
                           loop_sampler_mode=False, sampler_mode_method="sequential",
                           skip_sampler_modes=""):

        rng = random.Random(seed)

        # sampler list
        samp_all_raw = list(get_sampler_name_list())
        samp_skip = set(_parse_skip_list(skip_samplers))
        samp_all = [s for s in samp_all_raw if s not in samp_skip] or samp_all_raw

        # scheduler list
        sched_all_raw = list(get_res4lyf_scheduler_list())
        sched_skip = set(_parse_skip_list(skip_schedulers))
        sched_all = [s for s in sched_all_raw if s not in sched_skip] or sched_all_raw

        # reset states if asked
        if reset:
            RES4LYFComboLoop._samp_state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            RES4LYFComboLoop._samp_state["first_flag"].clear()
            RES4LYFComboLoop._sched_state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            RES4LYFComboLoop._sched_state["first_flag"].clear()
            RES4LYFComboLoop._mode_state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            RES4LYFComboLoop._mode_state["first_flag"].clear()

        _advance_counter(RES4LYFComboLoop._samp_state, sampler_loop_mode)
        _advance_counter(RES4LYFComboLoop._sched_state, scheduler_loop_mode)

        samp_step = RES4LYFComboLoop._samp_state["counters"][sampler_loop_mode]
        sched_step = RES4LYFComboLoop._sched_state["counters"][scheduler_loop_mode]

        i_samp = _select_index(len(samp_all), samp_step, sampler_loop_mode, rng)
        i_sched = _select_index(len(sched_all), sched_step, scheduler_loop_mode, rng)

        sampler = samp_all[i_samp]
        scheduler = sched_all[i_sched]

        # sampler_mode
        sampler_mode, i_mode, modes_total = _loop_sampler_mode(loop_sampler_mode, sampler_mode_method, seed, skip_sampler_modes, RES4LYFComboLoop._mode_state)

        # total combinations product
        total_combinations = len(samp_all) * len(sched_all) * (modes_total if loop_sampler_mode else 1)

        label = f"{sampler} | {scheduler} | {sampler_mode}"
        return (sampler, scheduler, sampler_mode, i_samp, i_sched, i_mode, total_combinations, label)

class DoubleIntRangeLoop:
    """
    Loop through combinations of two integer ranges (value1 x value2), sequentially.
    """
    _global_counter = 0
    _last_execution_id = None

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("value1", "value2", "current_index", "total_combinations", "current_combination")
    FUNCTION = "loop_ints"
    CATEGORY = "RES4LYF/Loop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1_start": ("INT",   {"default": 0,  "min": -0x7fffffffffffffff, "max": 0xffffffffffffffff}),
                "value1_end":   ("INT",   {"default": 5,  "min": -0x7fffffffffffffff, "max": 0xffffffffffffffff}),
                "value1_step":  ("INT",   {"default": 1,  "min": 1,                   "max": 0x7fffffff}),
                "value2_start": ("INT",   {"default": 0,  "min": -0x7fffffffffffffff, "max": 0xffffffffffffffff}),
                "value2_end":   ("INT",   {"default": 3,  "min": -0x7fffffffffffffff, "max": 0xffffffffffffffff}),
                "value2_step":  ("INT",   {"default": 1,  "min": 1,                   "max": 0x7fffffff}),
                "seed":         ("INT",   {"default": 0,  "min": 0,                   "max": 0xffffffffffffffff}),
                "reset":        ("BOOLEAN", {"default": False}),
            }
        }

    def loop_ints(self, value1_start, value1_end, value1_step,
                  value2_start, value2_end, value2_step, seed, reset=False):

        import threading, time

        v1_values = _gen_int_range(value1_start, value1_end, value1_step)
        v2_values = _gen_int_range(value2_start, value2_end, value2_step)

        total_combinations = len(v1_values) * len(v2_values)
        if total_combinations == 0:
            # graceful fallback
            return (value1_start, value2_start, 0, 0, "value1 ?, value2 ?")

        if reset:
            IntRangeLoop._global_counter = 0

        exec_id = f"{time.time()}_{threading.current_thread().ident}"
        if DoubleIntRangeLoop._last_execution_id != exec_id:
            DoubleIntRangeLoop._last_execution_id = exec_id
            if DoubleIntRangeLoop._global_counter > 0 or hasattr(self, "_first_call_done"):
                DoubleIntRangeLoop._global_counter += 1
            else:
                setattr(self, "_first_call_done", True)

        step = DoubleIntRangeLoop._global_counter
        index = step % total_combinations

        # map flat index -> (i1, i2)
        i1 = index // len(v2_values)
        i2 = index %  len(v2_values)

        selected_v1 = v1_values[i1]
        selected_v2 = v2_values[i2]
        label = f"value1 {selected_v1}, value2 {selected_v2}"

        return (selected_v1, selected_v2, index, total_combinations, label)

# ------------------------------
# FloatRangeLoop
# ------------------------------
class DoubleFloatRangeLoop:
    """
    Loop through combinations of two float ranges (value1 x value2), sequentially.
    """
    _global_counter = 0
    _last_execution_id = None

    RETURN_TYPES = ("FLOAT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("value1", "value2", "current_index", "total_combinations", "current_combination")
    FUNCTION = "loop_floats"
    CATEGORY = "RES4LYF/Loop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1_start": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "value1_end":   ("FLOAT", {"default": 5.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "value1_step":  ("FLOAT", {"default": 1.0, "min":  1e-6, "max": 1e6, "step": 0.01}),
                "value2_start": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "value2_end":   ("FLOAT", {"default": 3.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "value2_step":  ("FLOAT", {"default": 0.5, "min":  1e-6, "max": 1e6, "step": 0.01}),
                "seed":         ("INT",   {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset":        ("BOOLEAN", {"default": False}),
            }
        }

    def loop_floats(self, value1_start, value1_end, value1_step,
                    value2_start, value2_end, value2_step, seed, reset=False):

        import threading, time

        v1_values = _gen_float_range(value1_start, value1_end, value1_step, decimals=6)
        v2_values = _gen_float_range(value2_start, value2_end, value2_step, decimals=6)

        total_combinations = len(v1_values) * len(v2_values)
        if total_combinations == 0:
            return (value1_start, value2_start, 0, 0, "value1 ?, value2 ?")

        if reset:
            DoubleFloatRangeLoop._global_counter = 0

        exec_id = f"{time.time()}_{threading.current_thread().ident}"
        if DoubleFloatRangeLoop._last_execution_id != exec_id:
            DoubleFloatRangeLoop._last_execution_id = exec_id
            if DoubleFloatRangeLoop._global_counter > 0 or hasattr(self, "_first_call_done"):
                DoubleFloatRangeLoop._global_counter += 1
            else:
                setattr(self, "_first_call_done", True)

        step = DoubleFloatRangeLoop._global_counter
        index = step % total_combinations

        i1 = index // len(v2_values)
        i2 = index %  len(v2_values)

        selected_v1 = v1_values[i1]
        selected_v2 = v2_values[i2]
        # compact label without trailing zeros
        label = f"value1 {selected_v1}, value2 {selected_v2}"

        return (selected_v1, selected_v2, index, total_combinations, label)

class SingleIntLoop:
    """
    Loop through a single integer range, with sequential/random/ping_pong.
    """
    _state = {
        "counters":   {"sequential": 0, "random": 0, "ping_pong": 0},
        "last_exec":  {"sequential": None, "random": None, "ping_pong": None},
        "first_flag": set(),
    }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("value", "current_index", "total_combinations", "current_combination")
    FUNCTION = "loop_int_single"
    CATEGORY = "RES4LYF/Loop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode":       (["sequential", "random", "ping_pong"],),
                "start":      ("INT",   {"default": 0, "min": -0x7fffffffffffffff, "max": 0xffffffffffffffff}),
                "end":        ("INT",   {"default": 5, "min": -0x7fffffffffffffff, "max": 0xffffffffffffffff}),
                "step":       ("INT",   {"default": 1, "min": 1,                  "max": 0x7fffffff}),
                "seed":       ("INT",   {"default": 0, "min": 0,                  "max": 0xffffffffffffffff}),
                "reset":      ("BOOLEAN", {"default": False}),
            }
        }

    def loop_int_single(self, mode, start, end, step, seed, reset=False):
        import random

        values = _gen_int_range(start, end, step)
        total = len(values)
        if total == 0:
            return (start, 0, 0, "value ?")

        if reset:
            SingleIntLoop._state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            SingleIntLoop._state["first_flag"].clear()

        rng = random.Random(seed)
        _advance_counter(SingleIntLoop._state, mode)
        step_i = SingleIntLoop._state["counters"][mode]
        idx = _select_index(total, step_i, mode, rng)

        value = values[idx]
        label = f"value {value}"
        return (value, idx, total, label)


# ------------------------------
# Single Float Range Loop
# ------------------------------
class SingleFloatLoop:
    """
    Loop through a single float range, with sequential/random/ping_pong.
    """
    _state = {
        "counters":   {"sequential": 0, "random": 0, "ping_pong": 0},
        "last_exec":  {"sequential": None, "random": None, "ping_pong": None},
        "first_flag": set(),
    }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("value", "current_index", "total_combinations", "current_combination")
    FUNCTION = "loop_float_single"
    CATEGORY = "RES4LYF/Loop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode":       (["sequential", "random", "ping_pong"],),
                "start":      ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "end":        ("FLOAT", {"default": 5.0, "min": -1e9, "max": 1e9, "step": 0.01}),
                "step":       ("FLOAT", {"default": 1.0, "min":  1e-6, "max": 1e6, "step": 0.01}),
                "seed":       ("INT",   {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset":      ("BOOLEAN", {"default": False}),
            }
        }

    def loop_float_single(self, mode, start, end, step, seed, reset=False):
        import random

        values = _gen_float_range(start, end, step, decimals=6)
        total = len(values)
        if total == 0:
            return (start, 0, 0, "value ?")

        if reset:
            SingleFloatLoop._state["counters"] = {"sequential": 0, "random": 0, "ping_pong": 0}
            SingleFloatLoop._state["first_flag"].clear()

        rng = random.Random(seed)
        _advance_counter(SingleFloatLoop._state, mode)
        step_i = SingleFloatLoop._state["counters"][mode]
        idx = _select_index(total, step_i, mode, rng)

        value = values[idx]
        label = f"value {value}"
        return (value, idx, total, label)

# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "RES4LYFSchedulerLoop": RES4LYFSchedulerLoop,
    "RES4LYFSamplerLoop": RES4LYFSamplerLoop,
    "RES4LYFComboLoop": RES4LYFComboLoop,
    "DoubleIntRangeLoop": DoubleIntRangeLoop,
    "DoubleFloatRangeLoop": DoubleFloatRangeLoop,
    "SingleIntLoop": SingleIntLoop,
    "SingleFloatLoop": SingleFloatLoop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RES4LYFSchedulerLoop": "RES4LYF Scheduler Loop",
    "RES4LYFSamplerLoop": "RES4LYF Sampler Loop",
    "RES4LYFComboLoop": "RES4LYF Combo Loop (Sampler + Scheduler)",
    "DoubleIntRangeLoop": "Double Int Range Loop",
    "DoubleFloatRangeLoop": "Double Float Range Loop",
    "SingleIntLoop": "Single Int Range Loop",
    "SingleFloatLoop": "Single Float Range Loop",
}