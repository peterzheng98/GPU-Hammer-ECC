# GPU Rowhammer – ECC Error Trigger

Reproduces GPU Rowhammer attacks on HBM memory (NVIDIA Ampere/Hopper GPUs) to trigger ECC errors. Based on techniques from the [GPUHammer paper](https://github.com/sith-lab/gpuhammer).

## Requirements

- NVIDIA GPU with **sm_80+** (A100, H100, H200, or equivalent)
- CUDA Toolkit 11.0+
- NVIDIA driver with NVML support
- Linux (tested on Ubuntu 20.04+)

## Build

```bash
make
```

For a specific architecture only:

```bash
make CUDA_ARCH="80"       # A100 only
make CUDA_ARCH="90"       # H100 only
make CUDA_ARCH="80 90"    # Both (default)
```

## Quick Start

```bash
# Single run with default 24-sided pattern
./gpu_hammer

# Daemon mode: runs continuously, reports ECC errors, auto-restarts
./gpu_hammer --daemon

# Sweep through multiple patterns and data values
./gpu_hammer --daemon --sweep

# Verbose output with specific parameters
./gpu_hammer --pattern 24 --distance 4 --duration 256 --warps 8 --verbose --daemon
```

## How It Works

1. **Memory Allocation** – Allocates nearly all GPU memory via `cudaMalloc`.
2. **Bank Mapping** – Detects DRAM bank/row structure using timing-based row-buffer conflict analysis (or a user-provided stride).
3. **Synchronized Hammering** – Launches a k-warp, m-thread-per-warp CUDA kernel that rapidly activates aggressor rows using `discard.global.L2` + `ld.global.volatile`, with per-warp delay loops for REF synchronization.
4. **ECC Monitoring** – Polls NVML for correctable/uncorrectable memory error counters after each hammer.
5. **Bit-Flip Detection** – Verifies victim row data integrity to detect any flipped bytes.
6. **Daemon Loop** – In `--daemon` mode, repeats indefinitely, reporting and re-running to confirm reproducibility.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu ID` | 0 | GPU device index |
| `--pattern N` | 24 | N-sided aggressor pattern |
| `--distance D` | 4 | Logical row distance between aggressors |
| `--duration MS` | 128 | Hammer duration per position (ms) |
| `--warps K` | 8 | Number of warps for parallel hammering |
| `--threads M` | auto | Threads per warp (auto = N/K) |
| `--rounds R` | 1 | ACT rounds per sync window |
| `--sync-delay D` | auto | Delay iterations for REF sync (auto: 500 HBM, 200 GDDR) |
| `--stride BYTES` | auto | Manual same-bank stride (0 = auto-detect) |
| `--banks N` | 4 | Number of banks to test |
| `--reserve MB` | 256 | Memory to leave free (MB) |
| `--victim HEX` | 0xAA | Victim data pattern byte |
| `--aggressor HEX` | 0x55 | Aggressor data pattern byte |
| `--iterations N` | 0 | Max positions per bank (0 = all) |
| `--daemon` | off | Continuous mode with auto-restart |
| `--sweep` | off | Sweep through multiple patterns |
| `--cooldown SEC` | 5 | Pause between daemon runs |
| `--verbose` | off | Verbose output |

## ECC Setup

Check current ECC status:

```bash
nvidia-smi -q -d ECC
```

Enable ECC (requires reboot):

```bash
sudo nvidia-smi --ecc-config=1 -i 0
sudo reboot
```

Reset volatile ECC counters:

```bash
sudo nvidia-smi --reset-ecc-errors=volatile -i 0
```

## Architecture Notes

**HBM vs GDDR:** HBM GPUs (A100, H100) have on-die ECC that silently corrects single-bit errors. To trigger *observable* ECC events, the attack must cause disturbance exceeding on-die ECC correction capability, or cause enough single-bit corrections to increment the controller-level counters.

**Timing parameters:** The sync delay is auto-tuned based on GPU type. HBM has a longer tREFI (~3.9 µs) compared to GDDR6 (~1.4 µs), allowing more activations per refresh interval but requiring larger sync delays.

**Bank mapping:** HBM uses XOR-based bank interleaving with 256-byte granularity. The auto-detect mode tests common strides; if inconclusive, a full timing scan identifies same-bank addresses.

## Exit Codes

- `0` – ECC errors or bit flips were detected
- `1` – No errors detected (or aborted)
