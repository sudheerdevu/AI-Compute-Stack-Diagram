# GPU Memory Hierarchy

Visual guide to GPU memory hierarchy and optimization strategies.

## Memory Hierarchy Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          REGISTERS                                   │
│  Location: On-chip, per-thread        Latency: 0 cycles             │
│  Size: ~256 per thread                Bandwidth: Highest            │
│                                                                      │
│  ┌────┐┌────┐┌────┐┌────┐  Thread 0   ┌────┐┌────┐┌────┐┌────┐    │
│  │ r0 ││ r1 ││ r2 ││ r3 │  ...        │ r0 ││ r1 ││ r2 ││ r3 │    │
│  └────┘└────┘└────┘└────┘             └────┘└────┘└────┘└────┘    │
│                                                        Thread N      │
├──────────────────────────────────────────────────────────────────────┤
│                  LOCAL DATA SHARE (LDS) / SHARED MEMORY              │
│  Location: On-chip, per-block         Latency: ~20 cycles           │
│  Size: 64KB/128KB per CU              Bandwidth: ~10 TB/s           │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 Shared by all threads in block              │    │
│  │   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐      │    │
│  │   │Bank│Bank│Bank│Bank│Bank│Bank│Bank│Bank│...│Bank│      │    │
│  │   │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │   │ 31 │      │    │
│  │   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘      │    │
│  │                                                             │    │
│  │   32 banks, each 4 bytes wide, interleaved addressing      │    │
│  └─────────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────────┤
│                          L1 CACHE                                    │
│  Location: On-chip, per-CU            Latency: ~80 cycles           │
│  Size: 16-128KB                       Bandwidth: High               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Automatic caching of global memory              │    │
│  │              Cache line: 64-128 bytes                        │    │
│  │              Policies: Write-through, LRU                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────────┤
│                          L2 CACHE                                    │
│  Location: On-chip, shared            Latency: ~200 cycles          │
│  Size: 4-50MB                         Bandwidth: ~2-4 TB/s          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │         Shared across all CUs, acts as coherency point       │    │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
│  │   │ Slice 0  │ │ Slice 1  │ │ Slice 2  │ │ Slice N  │       │    │
│  │   └──────────┘ └──────────┘ └──────────┘ └──────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────────┤
│                     HIGH BANDWIDTH MEMORY (HBM)                      │
│  Location: Off-chip stacks            Latency: ~400 cycles          │
│  Size: 16-192GB                       Bandwidth: 500GB/s - 5TB/s    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐     │    │
│  │   │Stack│  │Stack│  │Stack│  │Stack│  │Stack│  │Stack│     │    │
│  │   │  0  │  │  1  │  │  2  │  │  3  │  │  4  │  │  5  │     │    │
│  │   └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘     │    │
│  │      └───────────────────┬───────────────────────┘          │    │
│  │                     Memory Controller                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Access Patterns

### Coalesced vs Non-Coalesced Access

```
COALESCED ACCESS (Good)
─────────────────────────────────────
Thread:    T0   T1   T2   T3   T4   T5   T6   T7
Address:   0    4    8    12   16   20   24   28

Memory: ┌────┬────┬────┬────┬────┬────┬────┬────┐
        │ 0  │ 4  │ 8  │ 12 │ 16 │ 20 │ 24 │ 28 │
        └────┴────┴────┴────┴────┴────┴────┴────┘
                    Single Transaction

NON-COALESCED ACCESS (Bad - 8x more transactions)
─────────────────────────────────────
Thread:    T0    T1    T2    T3    T4    T5    T6    T7
Address:   0     128   256   384   512   640   768   896

Memory: ┌────┐      ┌────┐      ┌────┐      ┌────┐
        │ 0  │      │128 │      │256 │      │384 │ ...
        └────┘      └────┘      └────┘      └────┘
           8 Separate Transactions
```

### Shared Memory Bank Conflicts

```
NO BANK CONFLICT (32 banks, 32 threads, 1 cycle)
──────────────────────────────────────────────────
Thread:  T0  T1  T2  T3  T4  ...  T31
Bank:    B0  B1  B2  B3  B4  ...  B31  ✓

2-WAY BANK CONFLICT (16 banks, 32 threads, 2 cycles)
──────────────────────────────────────────────────
Thread:  T0  T1  T2  ...  T16 T17 T18 ...
Bank:    B0  B1  B2  ...  B0  B1  B2  ...  ✗

BROADCAST (All threads same address, 1 cycle)
──────────────────────────────────────────────────
Thread:  T0  T1  T2  T3  T4  ...  T31
Bank:    B0  B0  B0  B0  B0  ...  B0   ✓ (special case)
```

## Memory Usage by AI Operation

```
MATRIX MULTIPLICATION (GEMM)
────────────────────────────────────────────────────────────
A [M×K] × B [K×N] = C [M×N]

Memory Access Pattern:
┌─────────────────────────────────────────────────┐
│         Global Memory (HBM)                     │
│  ┌───────┐  ┌───────┐  ┌───────┐               │
│  │   A   │  │   B   │  │   C   │               │
│  │ M×K   │  │ K×N   │  │ M×N   │               │
│  └───┬───┘  └───┬───┘  └───┬───┘               │
│      │          │          │                    │
│      │   Tile   │   Tile   │                    │
│      │   Load   │   Load   │                    │
│      ▼          ▼          ▼                    │
│  ┌─────────────────────────────────────────┐   │
│  │        Shared Memory (LDS)               │   │
│  │  ┌─────┐      ┌─────┐                    │   │
│  │  │A_tile│      │B_tile│                   │   │
│  │  │T×K' │      │K'×T  │                   │   │
│  │  └──┬──┘      └──┬──┘                    │   │
│  │     │  Compute   │                        │   │
│  │     └─────┬──────┘                        │   │
│  │           ▼                               │   │
│  │      ┌────────┐                           │   │
│  │      │C_accum │  (Registers)              │   │
│  │      │ T×T    │                           │   │
│  │      └────────┘                           │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘

Data Reuse:
- Each element of A_tile used T times
- Each element of B_tile used T times
- C_accum stays in registers
- Reduction in global memory traffic: ~T×
```

## Memory Bandwidth Comparison

```
╔════════════════════════════════════════════════════════════════╗
║                  MEMORY BANDWIDTH BY TYPE                       ║
╠════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Registers     █████████████████████████████████████  ~100 TB/s ║
║                                                                  ║
║  LDS/Shared    ██████████████████████████████        ~10-20 TB/s║
║                                                                  ║
║  L1 Cache      ████████████████████                  ~5 TB/s    ║
║                                                                  ║
║  L2 Cache      ██████████████                        ~2-4 TB/s  ║
║                                                                  ║
║  HBM3          ████████                              ~3-5 TB/s  ║
║                                                                  ║
║  HBM2e         ██████                                ~1.5-3 TB/s║
║                                                                  ║
║  GDDR6X        ████                                  ~1 TB/s    ║
║                                                                  ║
║  PCIe Gen5     ██                                    ~64 GB/s   ║
║                                                                  ║
║  PCIe Gen4     █                                     ~32 GB/s   ║
║                                                                  ║
╚════════════════════════════════════════════════════════════════╝
```

## Memory Optimization Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION STRATEGIES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. MAXIMIZE DATA REUSE                                         │
│     ├── Use shared memory for data accessed multiple times      │
│     ├── Keep working set in registers when possible             │
│     └── Tile algorithms to fit cache                            │
│                                                                  │
│  2. ENSURE COALESCED ACCESS                                     │
│     ├── Access contiguous memory addresses                      │
│     ├── Align data to cache line boundaries                     │
│     └── Consider data layout (AoS vs SoA)                       │
│                                                                  │
│  3. AVOID BANK CONFLICTS                                        │
│     ├── Pad shared memory arrays (+1 technique)                 │
│     ├── Use conflict-free access patterns                       │
│     └── Leverage broadcast for read-only data                   │
│                                                                  │
│  4. MINIMIZE HOST-DEVICE TRANSFERS                              │
│     ├── Keep data on GPU as long as possible                    │
│     ├── Overlap compute with async transfers                    │
│     └── Use pinned memory for faster PCIe transfers             │
│                                                                  │
│  5. RIGHT-SIZE PRECISION                                        │
│     ├── FP16/BF16 halves memory bandwidth requirements          │
│     ├── INT8 quarters bandwidth vs FP32                         │
│     └── Consider mixed precision approaches                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
