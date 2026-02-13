# AI Compute Stack Diagram 📊

Visual documentation of the complete AI compute stack, from hardware to frameworks.

## Overview

Understanding the full AI compute stack is essential for performance engineers. This repository provides clear, educational diagrams explaining how each layer works and interacts.

## Diagrams

### 1. Full Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  PyTorch │ TensorFlow │ JAX │ ONNX Runtime │ vLLM │ TensorRT   │
├─────────────────────────────────────────────────────────────────┤
│                      OPERATOR LAYER                             │
│         cuDNN │ MIOpen │ rocBLAS │ cuBLAS │ oneDNN             │
├─────────────────────────────────────────────────────────────────┤
│                      RUNTIME LAYER                              │
│           CUDA │ HIP │ OpenCL │ SYCL │ DirectML                │
├─────────────────────────────────────────────────────────────────┤
│                      DRIVER LAYER                               │
│              GPU Driver │ Kernel Module │ Firmware              │
├─────────────────────────────────────────────────────────────────┤
│                      HARDWARE LAYER                             │
│     NVIDIA GPU │ AMD GPU │ Intel GPU │ NPU │ TPU │ CPU         │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Inference Pipeline

See [diagrams/inference_pipeline.md](diagrams/inference_pipeline.md)

### 3. Memory Hierarchy

See [diagrams/memory_hierarchy.md](diagrams/memory_hierarchy.md)

### 4. GPU Architecture

See [diagrams/gpu_architecture.md](diagrams/gpu_architecture.md)

## Detailed Stack Layers

### Application Layer
Frameworks providing high-level APIs for model development and inference.

### Operator Layer  
Optimized implementations of core operations (GEMM, convolution, attention).

### Runtime Layer
Hardware abstraction and kernel launch infrastructure.

### Driver Layer
OS interface to hardware, memory management, scheduling.

### Hardware Layer
Physical compute units, memory controllers, interconnects.

## Use Cases

- **Education**: Teaching ML engineers about the compute stack
- **Debugging**: Understanding where performance issues originate
- **Architecture**: Designing efficient AI systems
- **Communication**: Explaining complex systems to stakeholders

## File Format

Diagrams are provided in multiple formats:
- **Mermaid** (`.md`): Renders in GitHub, VSCode
- **SVG** (`.svg`): High-quality vector graphics
- **ASCII** (`.txt`): Universal compatibility

## License

CC BY-SA 4.0 - Educational use encouraged

## Author

Sudheer Devu - AI Performance Engineer
