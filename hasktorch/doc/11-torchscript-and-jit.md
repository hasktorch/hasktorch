---
title: TorchScript and the JIT
---

# TorchScript and the JIT

Hasktorch executes eagerly: every tensor operation is one ATen call
that materializes its result. For a formula like the `iou` of the
previous chapter that means a dozen intermediate `n × n` tensors per
evaluation — correct, but memory-bandwidth-bound. PyTorch's answer to
this is fusion: compile chains of pointwise operations into a single
kernel. This chapter explains what of that machinery is reachable
from Hasktorch, and — importantly — why tracing alone will *not* make
your CPU code faster today.

## Tracing

`Torch.Script` binds TorchScript's tracer. Any Haskell function on
(untyped) tensors can be traced — including, unchanged, the
`a = Tensor` instantiation of a staged formula:

```haskell
import Torch.Script
import Torch.NN (forward)

m  <- trace "IoU" "forward" (\[a, b] -> return [iouOn a b]) exampleInputs
sm <- toScriptModule m
let IVTensor r = forward sm (map IVTensor inputs)
```

Tracing runs the function once while ATen records every dispatched
operation; the recorded graph can be saved (`saveScript`), loaded and
executed independently of the Haskell code that produced it. Tracing
the staged `iou` yields exactly the expected 15-node graph of
`aten::minimum`, `aten::maximum`, `aten::sub`, `aten::mul`,
`aten::add`, `aten::div`.

## Why it does not get faster on CPU

Measured on the `iou` formula at `n = 6000` (CPU):

| | ms/iter |
|---|---|
| eager | ≈ 86 |
| traced, default settings | ≈ 113 |

Traced execution is *slower*: the profiling executor adds overhead
and, by default, never fuses on CPU. Two internal switches control
this, exposed in `Torch.Internal.Unmanaged.Type.Module`:

```haskell
overrideCanFuseOnCPU 1       -- allow the fuser to take CPU graphs
setTensorExprFuserEnabled 1  -- enable the TensorExpr (NNC) fusion pass
```

With these on, the fuser does engage — it grabs the pointwise chain
and attempts to compile a fused kernel — and then fails at runtime
with:

```
LLVM Backend not found
```

The official libtorch binaries ship without NNC's LLVM code
generator. So CPU fusion via TorchScript is not a configuration away;
it requires a libtorch built with `USE_LLVM=ON`. This is a property
of the libtorch distribution, not of Hasktorch — upstream PyTorch
moved CPU fusion effort to `torch.compile`/Inductor, which is
Python-only.

The situation on CUDA is different: the CUDA fuser generates kernels
through NVRTC and has no LLVM dependency, so a traced module on GPU
can genuinely fuse. The switches above (`overrideCanFuseOnGPU`) apply
there as well.

## What to do instead, on CPU

Two things that *did* measurably help the eager path, both applied in
`Torch.Typed.Staged` and `Torch.Typed.Vision`:

1. **Cheaper interpreters.** `maxE`/`minE` are `Cond` class methods
   with a `whereE`-based default; the `Tensor` instance overrides
   them with native `aten::maximum`/`minimum` — one call instead of
   four. This roughly halved the op count of `iou` without touching
   any formula. Optimizing the interpreter rather than the programs
   is the point of writing element code against a class.

2. **Computing less.** `nms` stopped materializing the `n × n` IoU
   matrix and evaluates only the rows of boxes that are actually
   kept (`O(n)` memory, one broadcast evaluation per kept box) —
   a 7× end-to-end improvement at `n = 6000`.

The honest summary: eager Hasktorch reaches within a small factor of
native kernels when the operation count is kept down; closing the
rest needs either a CUDA device, an LLVM-enabled libtorch, or a
different backend behind the same polymorphic element code.
