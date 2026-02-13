# AI Compute Stack Layers Explained

## Layer 1: Applications
```
┌─────────────────────────────────────────────────────────────────┐
│                     AI APPLICATIONS                              │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│   NLP       │   Vision    │   Speech    │   Generative        │
│  ─────────  │  ─────────  │  ─────────  │  ─────────────      │
│  ChatGPT    │  Object     │  Whisper    │  Stable Diffusion   │
│  Claude     │  Detection  │  Text-to-   │  Midjourney         │
│  Llama      │  Image      │  Speech     │  DALL-E             │
│  BERT       │  Class.     │  ASR        │  GPT-4 Vision       │
└─────────────┴─────────────┴─────────────┴─────────────────────┘
```

## Layer 2: Frameworks
```
┌─────────────────────────────────────────────────────────────────┐
│                     DEEP LEARNING FRAMEWORKS                     │
├─────────────────────────────────────────────────────────────────┤
│  Training Frameworks         │  Inference Frameworks            │
│  ─────────────────          │  ────────────────────            │
│  • PyTorch                   │  • ONNX Runtime                  │
│  • TensorFlow                │  • TensorRT                      │
│  • JAX                       │  • MIGraphX                      │
│                              │  • vLLM                          │
├─────────────────────────────────────────────────────────────────┤
│  Distributed Training        │  Model Optimization              │
│  ─────────────────          │  ────────────────────            │
│  • DeepSpeed                 │  • Quantization                  │
│  • FSDP                      │  • Pruning                       │
│  • Megatron-LM               │  • Distillation                  │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 3: Compilers & Runtimes
```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPILER STACK                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  High-Level IR      →    Low-Level IR    →    Target Code       │
│    (Graph)               (Tensor)             (Kernel)          │
│                                                                  │
│  ┌─────────┐         ┌─────────┐         ┌─────────┐           │
│  │ TorchFX │   →    │  MLIR   │   →    │   LLVM  │   →  GPU   │
│  │   XLA   │         │  Triton │         │   SPIR  │     Code  │
│  │   TVM   │         │         │         │   GCN   │           │
│  └─────────┘         └─────────┘         └─────────┘           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     OPTIMIZATIONS                                │
│                                                                  │
│  • Operator Fusion (Conv + BN + ReLU → single kernel)           │
│  • Memory Planning (minimize allocations)                        │
│  • Layout Optimization (NCHW vs NHWC)                           │
│  • Constant Folding (pre-compute static values)                  │
│  • Dead Code Elimination                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 4: Kernel Libraries
```
┌─────────────────────────────────────────────────────────────────┐
│                     HIGH-PERFORMANCE LIBRARIES                   │
├─────────────┬─────────────────────────────────────────────────┤
│             │        AMD (ROCm)      │     NVIDIA (CUDA)       │
├─────────────┼─────────────────────────────────────────────────┤
│   BLAS      │   rocBLAS, hipBLAS     │     cuBLAS, cuBLASLt   │
│             │                         │                         │
│   DNN       │   MIOpen               │     cuDNN               │
│             │                         │                         │
│   FFT       │   rocFFT               │     cuFFT               │
│             │                         │                         │
│   Sparse    │   rocSPARSE            │     cuSPARSE            │
│             │                         │                         │
│   RNG       │   rocRAND              │     cuRAND              │
│             │                         │                         │
│   Comm      │   RCCL                 │     NCCL                │
│             │                         │                         │
│   Solver    │   rocSOLVER            │     cuSOLVER            │
└─────────────┴─────────────────────────────────────────────────┘
```

## Layer 5: Programming Model
```
┌─────────────────────────────────────────────────────────────────┐
│                     GPU PROGRAMMING MODEL                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│       Grid (NDRange)                                            │
│       ┌────────────────────────────────────────┐                │
│       │  Block 0,0    Block 0,1    Block 0,2   │                │
│       │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │                │
│       │  │ █ █ █ █ │  │ █ █ █ █ │  │ █ █ █ █ │ │                │
│       │  │ █ █ █ █ │  │ █ █ █ █ │  │ █ █ █ █ │ │                │
│       │  └─────────┘  └─────────┘  └─────────┘ │                │
│       │  Block 1,0    Block 1,1    Block 1,2   │                │
│       │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │                │
│       │  │ █ █ █ █ │  │ █ █ █ █ │  │ █ █ █ █ │ │                │
│       │  │ █ █ █ █ │  │ █ █ █ █ │  │ █ █ █ █ │ │                │
│       │  └─────────┘  └─────────┘  └─────────┘ │                │
│       └────────────────────────────────────────┘                │
│                                                                  │
│   HIP/CUDA Mapping:                                             │
│   • Grid     = Total work items                                 │
│   • Block    = Workgroup (shared memory scope)                  │
│   • Thread   = Single execution unit                            │
│   • Warp/Wave = SIMD execution group (32/64 threads)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 6: Hardware Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     GPU HARDWARE ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌─────────────────────┐                      │
│                    │    Command Engine   │                      │
│                    │    (Schedule Work)  │                      │
│                    └──────────┬──────────┘                      │
│                               │                                  │
│    ┌──────────────────────────┼──────────────────────────┐     │
│    │  Compute Units (CUs) / Streaming Multiprocessors (SMs)    │
│    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│    │  │ CU/SM 0  │  │ CU/SM 1  │  │ CU/SM 2  │  │ CU/SM N  │  │
│    │  │┌────────┐│  │┌────────┐│  │┌────────┐│  │┌────────┐│  │
│    │  ││ SIMD   ││  ││ SIMD   ││  ││ SIMD   ││  ││ SIMD   ││  │
│    │  ││ Units  ││  ││ Units  ││  ││ Units  ││  ││ Units  ││  │
│    │  │├────────┤│  │├────────┤│  │├────────┤│  │├────────┤│  │
│    │  ││ LDS/   ││  ││ LDS/   ││  ││ LDS/   ││  ││ LDS/   ││  │
│    │  ││ Shared ││  ││ Shared ││  ││ Shared ││  ││ Shared ││  │
│    │  │├────────┤│  │├────────┤│  │├────────┤│  │├────────┤│  │
│    │  ││Register││  ││Register││  ││Register││  ││Register││  │
│    │  │└────────┘│  │└────────┘│  │└────────┘│  │└────────┘│  │
│    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│    └───────────────────────────────────────────────────────────┘│
│                               │                                  │
│                    ┌──────────┴──────────┐                      │
│                    │   L2 Cache (Shared) │                      │
│                    └──────────┬──────────┘                      │
│                               │                                  │
│                    ┌──────────┴──────────┐                      │
│                    │  HBM/GDDR (Global)  │                      │
│                    │  High Bandwidth Mem │                      │
│                    └─────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Through Stack

```
User Application
       │
       ▼
┌─────────────────┐
│  model(input)   │  ← Python/Framework API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Graph Capture  │  ← Build computation graph
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Graph Optimize │  ← Fusion, layout, scheduling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kernel Select  │  ← Choose optimal implementations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kernel Launch  │  ← HIP/CUDA runtime
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Execution  │  ← Hardware compute
└────────┬────────┘
         │
         ▼
     Results
```
