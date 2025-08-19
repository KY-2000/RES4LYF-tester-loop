# RES4LYF Loop Helpers for ComfyUI

A comprehensive collection of loop nodes for ComfyUI that provides advanced iteration capabilities for samplers, schedulers, and numeric ranges. This custom node pack is designed to work seamlessly with the RES4LYF sampling system.

## 🙏 Special Thanks

**Special thanks to [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) for providing the incredible custom sampling and scheduling system for image generation.** This loop helper pack is built specifically to complement and enhance the RES4LYF workflow capabilities in ComfyUI.

## ✨ Features

### Core Loop Nodes
- **RES4LYF Scheduler Loop** - Iterate through available schedulers with intelligent fallbacks
- **RES4LYF Sampler Loop** - Cycle through Beta sampler list with comprehensive coverage
- **RES4LYF Combo Loop** - Combined sampler + scheduler + sampler mode iteration
- **Double Int Range Loop** - Loop through combinations of two integer ranges
- **Double Float Range Loop** - Loop through combinations of two float ranges  
- **Single Int Loop** - Loop through a single integer range with multiple modes
- **Single Float Loop** - Loop through a single float range with multiple modes

### Loop Modes
- **Sequential** - Step through values in order
- **Random** - Select values randomly (with seed control)
- **Ping Pong** - Bounce back and forth through the range

### Advanced Features
- **Sampler Mode Looping** - Toggle between `unsample`, `standard`, and `resample` modes
- **Skip Lists** - Exclude specific samplers, schedulers, or modes from loops
- **Label Outputs** - Human-readable combination descriptions
- **Reset Controls** - Reset loop counters on demand
- **Seed Support** - Reproducible random selection
- **Total Combinations** - Know exactly how many iterations are possible

## 📋 Prerequisites

### Required: RES4LYF Installation

**⚠️ IMPORTANT:** To get the complete list of samplers and schedulers (and avoid missing any), you **MUST** install RES4LYF first:

1. **Install RES4LYF** from the official repository:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/ClownsharkBatwing/RES4LYF.git
   ```

2. **Verify RES4LYF structure** - Ensure these files exist:
   ```
   ComfyUI/custom_nodes/RES4LYF/
   ├── helper.py
   └── beta/
       └── rk_coefficients_beta.py
   ```

3. **Install this loop helper pack** in the same custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone [your-repository-url] RES4LYF-Loop-Helpers
   ```

### Fallback Behavior

If RES4LYF is not installed or files are missing, the nodes will use built-in fallback lists:
- **Samplers**: 100+ fallback samplers across all categories (multistep, exponential, hybrid, linear, etc.)
- **Schedulers**: 11 fallback schedulers (simple, sgm_uniform, karras, exponential, etc.)

However, these fallbacks may not include the latest samplers and schedulers available in RES4LYF.

## 🚀 Usage

### Basic Scheduler Loop
```
RES4LYF Scheduler Loop
├── mode: sequential/random/ping_pong
├── seed: 0 (for random mode)
├── reset: false
└── skip_schedulers: "beta, normal" (optional)
```

**Outputs:**
- `scheduler` - Selected scheduler name
- `current_index` - Current position in the list
- `total_combinations` - Total possible combinations
- `current_combination` - Human-readable description
- `sampler_mode` - Current sampler mode (if looping enabled)
- `label` - Formatted combination string

### Advanced Combo Loop
```
RES4LYF Combo Loop
├── sampler_loop_mode: sequential
├── scheduler_loop_mode: random
├── seed: 42
├── skip_samplers: "none"
├── skip_schedulers: ""
├── loop_sampler_mode: true
├── sampler_mode_method: ping_pong
└── skip_sampler_modes: "unsample"
```

This will cycle through:
- All available samplers (sequentially)
- All available schedulers (randomly)
- Available sampler modes (ping-pong: standard ↔ resample)

### Numeric Range Loops

**Double Int Range:**
```
Double Int Range Loop
├── value1_start: 0, value1_end: 10, value1_step: 2
├── value2_start: 5, value2_end: 15, value2_step: 3
└── seed: 0
```
Produces combinations: (0,5), (0,8), (0,11), (0,14), (2,5), (2,8), ...

**Single Float Range:**
```
Single Float Loop
├── mode: ping_pong
├── start: 0.5, end: 2.0, step: 0.1
└── seed: 123
```

## 🔧 Node Details

### RES4LYF Scheduler Loop
- **Category**: RES4LYF/Loop
- **Function**: Loop through scheduler options with sampler mode support
- **Special**: Automatically detects available schedulers from RES4LYF installation

### RES4LYF Sampler Loop  
- **Category**: RES4LYF/Loop
- **Function**: Loop through Beta sampler list with sampler mode support
- **Special**: Supports 100+ samplers across all RES4LYF categories

### RES4LYF Combo Loop
- **Category**: RES4LYF/Loop  
- **Function**: Combined sampler + scheduler + mode iteration
- **Special**: Calculates total combinations as: samplers × schedulers × modes

### Range Loops
- **Double Loops**: Cartesian product of two ranges
- **Single Loops**: Individual range iteration with mode selection
- **Precision**: Float loops support 6 decimal places

## 📊 Sampler Categories

The nodes support all RES4LYF sampler categories:
- **Multistep**: res_2m, res_3m, dpmpp_2m, dpmpp_3m, abnorsett, deis
- **Exponential**: res_2s through res_16s, etdrk variants, dpmpp, lawson, ddim
- **Hybrid**: pec423, pec433, abnorsett, lawson variants
- **Linear**: ralston, midpoint, heun, runge-kutta variants, bogacki-shampine
- **Diag Implicit**: irk_exp_diag, kraaijevanger_spijker, qin_zhang, pareschi_russo
- **Fully Implicit**: gauss-legendre, radau variants, lobatto variants

## 🛠️ Installation Troubleshooting

### Missing Samplers/Schedulers?
1. Verify RES4LYF is properly installed
2. Check that `helper.py` and `beta/rk_coefficients_beta.py` exist
3. Restart ComfyUI after installing RES4LYF
4. Check ComfyUI console for import errors

### Node Not Appearing?
1. Ensure both RES4LYF and this pack are in `custom_nodes/`
2. Restart ComfyUI completely
3. Check for Python errors in the console
4. Verify file permissions

## 🤝 Contributing

This project builds upon the excellent work of RES4LYF. Contributions are welcome!

## 📄 License

Please respect the licensing terms of the RES4LYF project that this pack depends on.

---

**Powered by [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF)** - Advanced sampling and scheduling for ComfyUI