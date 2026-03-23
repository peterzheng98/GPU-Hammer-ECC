# GPU Rowhammer – ECC Error Trigger

Reproduces GPU Rowhammer attacks on HBM and GDDR memory (NVIDIA Ampere/Hopper/Blackwell GPUs) to trigger ECC errors. Based on techniques from the [GPUHammer paper](https://github.com/sith-lab/gpuhammer).

## Requirements

- NVIDIA GPU with **sm_80+** (A100, H100, H200, B100, A6000, RTX 30/40xx, etc.)
- CUDA Toolkit 11.0+
- NVIDIA driver with NVML support
- Linux (tested on Ubuntu 20.04+)

## Build

```bash
make
```

For a specific architecture only:

```bash
make CUDA_ARCH="80"       # A100 / A6000
make CUDA_ARCH="90"       # H100 / H200
make CUDA_ARCH="80 90"    # Both (default)
```

## Quick Start

```bash
# Run on all GPUs with default 24-sided pattern
./gpu_hammer

# Run on a specific GPU
./gpu_hammer --gpu 0

# Run on selected GPUs
./gpu_hammer --gpu 0,1,3

# Daemon mode with sweep on all GPUs
./gpu_hammer --daemon --sweep

# Verbose with custom parameters
./gpu_hammer --gpu 0 --pattern 24 --distance 4 --duration 256 --verbose --daemon
```

## How It Works

1. **Memory Type Detection** – Identifies HBM2e, HBM3, HBM3e, GDDR6, or GDDR6X based on the GPU model. Configures timing (tREFI, tRC), bank/channel topology, sync delays, and stride hints from the memory profile.
2. **HBM 3D Structure Awareness** – For HBM GPUs, the tool models the 3D stacked architecture: die stacks, pseudo-channels per stack, bank groups per channel. Bank mapping tags each bank with its estimated stack and channel, ensuring hammer patterns target rows within the same pseudo-channel where Rowhammer coupling occurs.
3. **Multi-GPU Parallel Execution** – By default, launches one worker thread per GPU. Each thread independently manages its own CUDA context, memory allocation, bank mapping, and hammer campaigns.
4. **Bank Mapping** – Detects DRAM bank/row structure using timing-based row-buffer conflict analysis. Stride candidates are tailored to the memory type (e.g., 32 KB for HBM2e, 64 KB for HBM3).
5. **Synchronized Hammering** – Launches a k-warp, m-thread-per-warp CUDA kernel that rapidly activates aggressor rows using `discard.global.L2` + `ld.global.volatile`, with per-warp delay loops sized to align with the memory type's tREFI for REF synchronization.
6. **ECC Monitoring** – Polls NVML for correctable/uncorrectable memory error counters after each hammer position.
7. **Logging** – All output goes to both stdout (with GPU prefix) and a timestamped log file for post-mortem analysis.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu ID\|all\|0,1,2` | all | GPU selection |
| `--pattern N` | 24 | N-sided aggressor pattern |
| `--distance D` | 4 | Logical row distance between aggressors |
| `--duration MS` | 128 | Hammer duration per position (ms) |
| `--warps K` | 8 | Number of warps for parallel hammering |
| `--threads M` | auto | Threads per warp (auto = N/K) |
| `--rounds R` | auto | ACT rounds per sync window (auto from memory profile) |
| `--sync-delay D` | auto | Delay iterations for REF sync (auto from memory profile) |
| `--stride BYTES` | auto | Manual same-bank stride (0 = auto-detect) |
| `--banks N` | auto | Number of banks to test (auto from memory profile) |
| `--reserve MB` | 256 | Memory to leave free (MB) |
| `--victim HEX` | 0xAA | Victim data pattern byte |
| `--aggressor HEX` | 0x55 | Aggressor data pattern byte |
| `--iterations N` | 0 | Max positions per bank (0 = all) |
| `--daemon` | off | Continuous mode with auto-restart |
| `--sweep` | off | Sweep through multiple patterns |
| `--cooldown SEC` | 5 | Pause between daemon runs |
| `--log FILE` | auto | Log file path (default: `gpu_hammer_<timestamp>.log`) |
| `--verbose` | off | Verbose output |

## Memory Type Detection

The tool explicitly identifies the memory technology and uses it to configure all parameters:

| GPU | Memory | Stacks | Channels | Banks | tREFI | tRC |
|-----|--------|--------|----------|-------|-------|-----|
| A100 | HBM2e | 5 | 40 | 640 | 3.9 µs | 46 ns |
| H100/H800 | HBM3 | 5 | 80 | 1280 | 3.9 µs | 36 ns |
| H200 | HBM3e | 6 | 96 | 1536 | 3.9 µs | 36 ns |
| B100/B200 | HBM3e | 8 | 128 | 2048 | 3.9 µs | 36 ns |
| A6000/RTX 30xx | GDDR6 | — | varies | varies | 1.4 µs | 45 ns |
| RTX 40xx/L40 | GDDR6X | — | varies | varies | 1.4 µs | 45 ns |

## ECC Setup

```bash
# Check current ECC status
nvidia-smi -q -d ECC

# Enable ECC (requires reboot)
sudo nvidia-smi --ecc-config=1 -i 0
sudo reboot

# Reset volatile ECC counters
sudo nvidia-smi --reset-ecc-errors=volatile -i 0
```

## Log File

Every run produces a log file (default: `gpu_hammer_YYYYMMDD_HHMMSS.log`) containing timestamped entries for:
- GPU identification and memory profile
- Bank mapping results (with stack/channel tags for HBM)
- Campaign parameters and progress
- ECC error events with counter diffs
- Bit-flip detections with byte-level detail
- Per-GPU and aggregate summary

## Architecture Notes

**HBM 3D Structure:** HBM memory is organized as vertically stacked DRAM dies connected via through-silicon vias (TSVs). Each stack contains multiple pseudo-channels (8 for HBM2e, 16 for HBM3), and each channel has its own bank array. Rowhammer coupling occurs within a single channel's bank, so the tool concentrates all hammer warps on addresses mapping to the same pseudo-channel. Center dies in the stack run hotter, potentially increasing vulnerability.

**HBM vs GDDR:** HBM GPUs (A100, H100) have on-die ECC that silently corrects single-bit errors. To trigger observable ECC events, the attack must cause disturbance exceeding on-die ECC correction capability. GDDR6/6X GPUs may lack on-die ECC, making bit-flips directly observable.

**Timing Parameters:** Sync delay and rounds are auto-tuned from the memory profile. HBM has a longer tREFI (~3.9 µs) vs GDDR6 (~1.4 µs), and HBM3 has faster tRC (36 ns vs 46 ns), fitting more activations per refresh interval.

## Exit Codes

- `0` – ECC errors or bit flips were detected
- `1` – No errors detected (or aborted)
