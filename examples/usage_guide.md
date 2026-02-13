# Example: Using AI Compute Stack Diagrams

This folder contains example usage scenarios for the diagrams.

## Use Case 1: Teach Memory Hierarchy

When explaining GPU memory to a new engineer:

1. Start with [memory_hierarchy.md](../diagrams/memory_hierarchy.md)
2. Explain registers → shared → global flow
3. Show concrete latency numbers
4. Discuss coalescing with diagram examples

## Use Case 2: Debug Performance Issues

When a kernel is slow:

1. Check [inference_pipeline.md](../diagrams/inference_pipeline.md)
2. Identify which layer has the bottleneck
3. Use [gpu_architecture.md](../diagrams/gpu_architecture.md) to understand hardware limits
4. Profile and correlate with diagram predictions

## Use Case 3: Architecture Design Reviews

When designing a new inference system:

1. Walk through full stack diagram in README
2. Discuss which layers are customizable
3. Identify integration points
4. Plan profiling strategy

## Tips for Presentations

- Use ASCII diagrams for terminal/text environments
- Export to SVG for high-quality slides
- Mermaid renders nicely in GitHub/Markdown viewers
- Customize colors for specific audiences
