# Data Flow in AI Compute Stack

## Inference Data Flow

This document illustrates how data flows through the AI compute stack during inference.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATA                                      │
│                                                                             │
│     ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │
│     │   Text       │      │   Image      │      │   Audio      │           │
│     │  "Hello..."  │      │  [H,W,C]     │      │  [Samples]   │           │
│     └──────────────┘      └──────────────┘      └──────────────┘           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING                                      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  • Tokenization (text)      • Resize/Normalize (images)              │ │
│  │  • Padding/Truncation       • Feature extraction                      │ │
│  │  • Batch collation          • Data type conversion                    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Output: Tensors ready for model                                            │
│  Text: [batch, seq_len] int64                                               │
│  Image: [batch, C, H, W] float32                                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOST → DEVICE TRANSFER                               │
│                                                                             │
│  ┌─────────────────────┐         ┌─────────────────────────────────────┐   │
│  │   Host Memory       │  PCIe   │         GPU Memory                  │   │
│  │   (RAM)            │ ──────► │         (VRAM)                      │   │
│  │                     │  16-64  │                                     │   │
│  │   Input Tensors    │  GB/s   │   Device Tensors                   │   │
│  └─────────────────────┘         └─────────────────────────────────────┘   │
│                                                                             │
│  Optimizations:                                                             │
│  • Pinned (page-locked) memory                                              │
│  • Async transfers with streams                                             │
│  • Double buffering                                                         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL EXECUTION                                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      Execution Graph                                   │ │
│  │                                                                        │ │
│  │   Input ──► [Embedding] ──► [Attention] ──► [FFN] ──► [Norm] ──► Out │ │
│  │                  │              │             │           │            │ │
│  │                  ▼              ▼             ▼           ▼            │ │
│  │              Kernel 1      Kernel 2      Kernel 3    Kernel 4        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Memory Hierarchy During Execution:                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   Registers (fastest) ◄──► Shared/LDS ◄──► L2 Cache ◄──► HBM/GDDR │   │
│  │        ~TB/s                 ~TB/s         ~TB/s         ~1TB/s     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ATTENTION COMPUTATION (LLM)                            │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │   Q ────────┐                                                          │ │
│  │             │                                                          │ │
│  │   K ────────┼──► QK^T/√d ──► Softmax ──► × V ──► Output              │ │
│  │             │                   │                                      │ │
│  │   V ────────┘                   │                                      │ │
│  │                                 ▼                                      │ │
│  │                          Attention Weights                            │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Optimizations:                                                             │
│  • Flash Attention: Fused computation, tiled for memory efficiency          │
│  • KV Cache: Store K,V for autoregressive generation                        │
│  • PagedAttention: Virtual memory for KV cache (vLLM)                       │
│  • Multi-Query/Grouped-Query Attention: Reduced KV heads                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEVICE → HOST TRANSFER                               │
│                                                                             │
│  ┌─────────────────────────────────────┐         ┌─────────────────────┐   │
│  │         GPU Memory                  │  PCIe   │   Host Memory       │   │
│  │                                     │ ──────► │   (RAM)            │   │
│  │   Output Logits                    │         │                     │   │
│  │   [batch, vocab_size]              │         │   Result Tensors   │   │
│  └─────────────────────────────────────┘         └─────────────────────┘   │
│                                                                             │
│  For LLMs: Usually only transfer top-k logits or sampled token             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          POSTPROCESSING                                      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  • Argmax / Sampling (LLM)      • Decode tokens → text               │ │
│  │  • NMS / Box decoding (Detection)• Mask decoding (Segmentation)      │ │
│  │  • Softmax → Probabilities      • Label mapping                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                          │
│                                                                             │
│     ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │
│     │   Text       │      │  Detections  │      │   Class      │           │
│     │  Response    │      │  [Boxes]     │      │ Probabilities│           │
│     └──────────────┘      └──────────────┘      └──────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Kernel Execution Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE KERNEL EXECUTION                                   │
│                                                                             │
│  GPU Block Grid (example: 128 blocks × 256 threads)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ... ┌──────┐         │   │
│  │  │Block │ │Block │ │Block │ │Block │ │Block │     │Block │         │   │
│  │  │  0   │ │  1   │ │  2   │ │  3   │ │  4   │     │ 127  │         │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     └──────┘         │   │
│  │                                                                      │   │
│  │  Each block contains:                                                │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  Warp 0   │  Warp 1   │  Warp 2   │  ...  │  Warp 7         │   │   │
│  │  │ (32 thds) │ (32 thds) │ (32 thds) │       │ (32 thds)       │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Execution Timeline:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  SM 0: [Blk0][Blk8 ][Blk16]...                                     │   │
│  │  SM 1: [Blk1][Blk9 ][Blk17]...                                     │   │
│  │  SM 2: [Blk2][Blk10][Blk18]...                                     │   │
│  │  ...                                                                │   │
│  │  SM 7: [Blk7][Blk15][Blk23]...                                     │   │
│  │                                                                      │   │
│  │  ─────────────────────────────────────────────────────► Time       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## LLM Autoregressive Generation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTOREGRESSIVE GENERATION                                 │
│                                                                             │
│  Prefill Phase (process prompt):                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Input: "What is AI?" → [Token1, Token2, Token3]                    │   │
│  │                                                                      │   │
│  │  Process all tokens in parallel:                                     │   │
│  │  ┌────┐ ┌────┐ ┌────┐                                               │   │
│  │  │ T1 │ │ T2 │ │ T3 │  ──► Full attention ──► Build KV Cache       │   │
│  │  └────┘ └────┘ └────┘                                               │   │
│  │                                                                      │   │
│  │  KV Cache populated for all prompt positions                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Decode Phase (generate tokens one by one):                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Step 1:  T3 ──► Model ──► T4 ("Artificial")                       │   │
│  │                   │                                                  │   │
│  │                   ▼                                                  │   │
│  │            [Use cached K,V for T1,T2,T3]                            │   │
│  │            [Compute new K,V for T4]                                 │   │
│  │            [Append to cache]                                        │   │
│  │                                                                      │   │
│  │  Step 2:  T4 ──► Model ──► T5 ("Intelligence")                     │   │
│  │  Step 3:  T5 ──► Model ──► T6 ("is")                               │   │
│  │  ...                                                                 │   │
│  │  Step N:  TN ──► Model ──► <EOS>                                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: "Artificial Intelligence is a field of computer science..."       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Batching Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BATCHING STRATEGIES                                   │
│                                                                             │
│  Static Batching (inefficient):                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Request 1: [████████████]                                          │   │
│  │  Request 2: [████████████████████]                                  │   │
│  │  Request 3: [████]                                                   │   │
│  │                   ▲                                                  │   │
│  │                   └── Wait for longest request (wasted compute)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Continuous Batching (efficient):                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Time ──►                                                           │   │
│  │  ┌───────────────┬───────────────┬───────────────┐                 │   │
│  │  │ R1 R2 R3      │ R2 R4 R5      │ R4 R5 R6 R7   │                 │   │
│  │  └───────────────┴───────────────┴───────────────┘                 │   │
│  │         │              │               │                            │   │
│  │         R1,R3 done     R2 done         Continuous...               │   │
│  │         New R4,R5      New R6,R7                                    │   │
│  │         join batch     join batch                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Critical Performance Metrics

| Layer | Key Metrics | Typical Bottlenecks |
|-------|-------------|---------------------|
| Host→Device | PCIe throughput, latency | Small transfers, synchronous copies |
| Model Execution | FLOPS utilization, memory bandwidth | Kernel launch overhead, low occupancy |
| Attention | Memory bound, QKV computation | Sequence length scaling, KV cache |
| Device→Host | Transfer size | Full tensor copies |
| Postprocessing | CPU utilization | Blocking operations |
