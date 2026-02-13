# AI Compute Stack Architecture

## Overview

This document provides a comprehensive architectural overview of the modern AI compute stack, from hardware to applications.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                                 │
│                                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   ChatGPT  │  │ Stable     │  │   Copilot  │  │   Custom   │            │
│  │   Claude   │  │ Diffusion  │  │   Codex    │  │   Apps     │            │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SERVING & INFERENCE                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Inference Servers                                │   │
│  │   • NVIDIA Triton    • vLLM           • TensorRT-LLM               │   │
│  │   • BentoML          • Text Generation Inference (TGI)             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Optimization Techniques                          │   │
│  │   • Continuous Batching    • PagedAttention    • Speculative Decode │   │
│  │   • KV Cache Optimization  • Flash Attention   • Tensor Parallelism │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FRAMEWORK LAYER                                     │
│                                                                             │
│  ┌───────────────────────────┐  ┌───────────────────────────┐              │
│  │      Training             │  │      Inference            │              │
│  │  • PyTorch               │  │  • ONNX Runtime           │              │
│  │  • TensorFlow            │  │  • TensorRT               │              │
│  │  • JAX                   │  │  • OpenVINO               │              │
│  │  • Megatron-LM           │  │  • DirectML               │              │
│  │  • DeepSpeed             │  │  • MIGraphX               │              │
│  └───────────────────────────┘  └───────────────────────────┘              │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Model Exchange Format                             │ │
│  │                          ONNX                                         │ │
│  │   • Graph Representation  • Operators  • Quantization  • Optimizers  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPILER LAYER                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Graph Compilers                                  │   │
│  │   • XLA (TensorFlow)     • TorchScript     • TVM                   │   │
│  │   • Triton              • MLIR             • IREE                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Optimizations                                    │   │
│  │   • Operator Fusion   • Memory Planning   • Kernel Selection       │   │
│  │   • Layout Transform  • Quantization      • Tiling                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KERNEL/LIBRARY LAYER                                │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │    NVIDIA Stack     │  │     AMD Stack       │  │   Intel Stack       │ │
│  │                     │  │                     │  │                     │ │
│  │  • cuBLAS          │  │  • rocBLAS          │  │  • oneMKL           │ │
│  │  • cuDNN           │  │  • MIOpen           │  │  • oneDNN           │ │
│  │  • CUTLASS         │  │  • hipBLAS          │  │  • Level Zero       │ │
│  │  • cuSPARSE        │  │  • rocSPARSE        │  │                     │ │
│  │  • cuRAND          │  │  • rocRAND          │  │                     │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RUNTIME LAYER                                       │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │       CUDA          │  │        HIP          │  │       SYCL          │ │
│  │                     │  │                     │  │                     │ │
│  │  • Memory Mgmt     │  │  • ROCm Runtime     │  │  • oneAPI           │ │
│  │  • Stream/Events   │  │  • HSA Runtime      │  │  • DPC++            │ │
│  │  • Graph Capture   │  │  • HIP Graph        │  │  • Level Zero       │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Cross-Platform                                    │ │
│  │   • OpenCL           • Vulkan Compute        • DirectML              │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DRIVER LAYER                                        │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │   NVIDIA Driver     │  │   AMD Driver        │  │   Intel Driver      │ │
│  │                     │  │                     │  │                     │ │
│  │  • nvidia.ko       │  │  • amdgpu.ko        │  │  • i915.ko          │ │
│  │  • libnvidia-ml    │  │  • libdrm_amdgpu    │  │  • xe.ko            │ │
│  │  • NVML            │  │  • ROCm SMI         │  │                     │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HARDWARE LAYER                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GPU Architecture                                 │   │
│  │                                                                      │   │
│  │   ┌────────────────────────────────────────────────────────────┐    │   │
│  │   │  Streaming Multiprocessors / Compute Units                 │    │   │
│  │   │                                                            │    │   │
│  │   │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │    │   │
│  │   │  │Tensor│ │Tensor│ │Tensor│ │Tensor│ │INT8  │ │  FP  │   │    │   │
│  │   │  │ Core │ │ Core │ │ Core │ │ Core │ │ Unit │ │ Unit │   │    │   │
│  │   │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘   │    │   │
│  │   │                                                            │    │   │
│  │   │  ┌────────────────────────────────────────────────────┐   │    │   │
│  │   │  │              Shared Memory / LDS                    │   │    │   │
│  │   │  └────────────────────────────────────────────────────┘   │    │   │
│  │   └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │   ┌────────────────────────────────────────────────────────────┐    │   │
│  │   │                    L2 Cache                                │    │   │
│  │   └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                      │   │
│  │   ┌────────────────────────────────────────────────────────────┐    │   │
│  │   │              High Bandwidth Memory (HBM)                   │    │   │
│  │   │                   / GDDR6 / GDDR6X                         │    │   │
│  │   └────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │   NVIDIA              │  │   AMD                │  │   Intel             │ │
│  │                       │  │                       │  │                     │ │
│  │  H100 / A100 / L4    │  │  MI300X / MI250      │  │  Max / Flex        │ │
│  │  RTX 4090 / 5090     │  │  RX 7900 XTX         │  │  Arc A770          │ │
│  │                       │  │  Radeon PRO         │  │                     │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Layer Details

### Application Layer
End-user applications consuming AI capabilities:
- **LLM Applications**: ChatGPT, Claude, Copilot, custom chatbots
- **Vision**: Image generation (Stable Diffusion), recognition
- **Multi-Modal**: Vision-language models like GPT-4V

### Serving & Inference Layer
Production serving infrastructure:
- **NVIDIA Triton**: Multi-framework serving
- **vLLM**: LLM optimized with PagedAttention
- **TGI**: Hugging Face's optimized inference

### Framework Layer
Training and inference frameworks:
- **Training**: PyTorch (dominant), TensorFlow, JAX
- **Inference**: ONNX Runtime (cross-platform), TensorRT (NVIDIA optimized)

### Compiler Layer
Graph-level optimizations:
- **XLA**: TensorFlow/JAX compiler
- **TVM**: Universal tensor compiler
- **Triton**: Python-based kernel compiler

### Kernel/Library Layer
Optimized compute kernels:
- **NVIDIA**: cuBLAS, cuDNN, CUTLASS
- **AMD**: rocBLAS, MIOpen (ROCm ecosystem)
- **Intel**: oneMKL, oneDNN

### Runtime Layer
GPU programming models:
- **CUDA**: NVIDIA proprietary
- **HIP**: AMD (CUDA-like API)
- **SYCL**: Cross-platform C++

### Driver Layer
Kernel-mode drivers and management:
- Hardware abstraction
- Memory management
- Multi-GPU coordination

### Hardware Layer
Physical GPU architectures:
- Tensor Cores / Matrix Units
- High-bandwidth memory
- Interconnects (NVLink, Infinity Fabric)
