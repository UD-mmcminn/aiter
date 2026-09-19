# gfx908 follow-on performance roadmap

Status: deferred follow-on work. This document records performance goals without
expanding the current gfx908 correctness and compatibility baseline.

## Hardware target

- Two hives with four AMD Instinct MI100 GPUs each.
- Measure and record the intra-hive and inter-hive GPU links, host NUMA placement,
  peer-to-peer bandwidth, and collective bandwidth before selecting an 8-GPU
  parallel strategy.
- Treat four-GPU hive-local execution and eight-GPU node execution as separate
  targets until measurements show that crossing the hive boundary is beneficial.

## Workload goals

### vLLM serving

Improve DeepSeek 4.1 FLASH serving on MI100 through AITER and vLLM. Confirm the
exact upstream model identifier, revision, and serving configuration when this
phase begins.

Track at least:

- prefill and decode throughput;
- time to first token and inter-token latency, including tail latency;
- useful batch size and context length;
- peak memory per GPU and KV-cache capacity;
- four-GPU and eight-GPU scaling efficiency; and
- output correctness against an untuned baseline.

### Training

Improve training throughput and reduce memory pressure for
`Qwen/Qwen3.8-27B`. Confirm the exact model revision and training recipe when
this phase begins.

Track at least:

- tokens per second and step time;
- maximum stable microbatch and sequence length;
- peak allocated and reserved GPU memory;
- activation, optimizer-state, gradient, and temporary-workspace memory;
- compute/communication overlap and scaling efficiency; and
- loss and gradient parity against an untuned baseline.

## Execution plan

### 1. Establish workload baselines

- Pin the ROCm, PyTorch, AITER, vLLM, model, and framework revisions.
- Capture representative prompt, sequence, batch, expert-routing, and GEMM
  shapes rather than optimizing synthetic shapes by assumption.
- Profile kernel time, launch gaps, memory bandwidth, occupancy, workspace
  allocation, host overhead, and collectives.
- Produce a ranked bottleneck list separately for serving prefill, serving
  decode, and training forward/backward/optimizer phases.

### 2. Optimize the critical single-GPU kernels

Choose targets from profiles. Likely families include attention, GEMM, MoE,
normalization, RoPE/cache handling, and training backward kernels, but none are
preselected without workload evidence.

Apply the least costly effective optimization first:

1. tune existing CK, Triton, FlyDSL, or HIP launch configurations;
2. remove avoidable conversions, intermediates, launches, and workspaces;
3. fuse producer/consumer operations where the public contract permits it;
4. specialize source kernels for common MI100 shapes; and
5. write or port gfx908 ISA/code objects only for stable, dominant bottlenecks
   where source-generated kernels leave meaningful performance available.

ISA work should follow the inspection and validation workflow in
[`isa_kernel_optimization.md`](isa_kernel_optimization.md). Every tuned kernel
needs a portable correctness fallback and shape/architecture gating.

### 3. Reduce training memory pressure

- Attribute peak memory before changing kernels.
- Prioritize fused operations and shorter-lived workspaces that preserve model
  numerics.
- Evaluate attention backward memory, activation recomputation boundaries,
  gradient accumulation, optimizer state, and communication buffers as distinct
  contributors.
- Keep framework-level changes separate from AITER kernel changes so their
  benefits can be measured and reviewed independently.

### 4. Optimize multi-GPU execution

- Establish peer-to-peer and collective bandwidth/latency baselines for one
  hive and across both hives.
- Profile the actual vLLM and training parallel strategies before modifying
  communication kernels.
- Evaluate tensor, pipeline, data, and expert parallelism according to the
  model workload and measured topology.
- Investigate fused collectives, reduce-scatter/all-gather substitutions,
  communication/compute overlap, persistent communication resources, and NUMA
  placement where profiles justify them.
- Preserve a four-GPU hive-local mode when the inter-hive path makes an eight-GPU
  configuration slower or less predictable.

### 5. Validate and package

- Compare every optimization with the correctness baseline.
- Run odd, boundary, and production-trace shapes, not only the tuned shape.
- Record cold-build and warm-cache behavior.
- Keep general correctness fixes, gfx908 enablement, workload tuning, ISA code
  objects, and multi-GPU changes in separately reviewable commits and PRs.

## Completion criteria for the follow-on program

The program is successful when it produces reproducible workload benchmarks,
a profile-backed set of optimized kernels/configurations, quantified memory
improvements, and measured four-/eight-GPU scaling. A large checked-in gfx908
code-object inventory is not itself a success criterion.

