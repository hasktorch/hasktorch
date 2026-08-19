---
title: Saving and Loading
---

# Saving and Loading

A trained model that dies with its process was never trained. This
chapter covers Hasktorch's serialization story: checkpointing models
from Haskell, and — since the weights are ordinary libtorch tensors —
exchanging them with PyTorch in both directions.

```haskell top hide
{-# LANGUAGE ScopedTypeVariables #-}
import Inliterate.Import (AskInliterate)
```

```haskell top
import Torch
import Torch.Script (IValue (..))
import Torch.Serialize
```

```haskell top hide
instance AskInliterate Tensor
instance AskInliterate IValue
instance AskInliterate Bool
```

## Checkpointing a model

`Torch.Serialize.saveParams` writes any `Parameterized` model — the
same typeclass the optimizers flatten with — and `loadParams` reads
one back. Loading is functional: you give it a model value of the
right shape to serve as the skeleton, and it returns a *new* value
with the stored parameters in place:

```haskell do
model <- sample (LinearSpec 3 1)
saveParams model "/tmp/linear-checkpoint.pt"

fresh <- sample (LinearSpec 3 1)
restored <- loadParams fresh "/tmp/linear-checkpoint.pt"
```

The freshly sampled model disagrees with the original, the restored
one is exact:

```haskell do
let probe = asTensor [[1, 2, 3 :: Float]]
```

```haskell eval
[linear model probe, linear fresh probe, linear restored probe]
```

That is the whole checkpointing API. The file holds the flattened
parameter list, so it does not remember your record's field names —
which is fine for save-and-resume with the same code, and is exactly
the underlying `save`/`load` pair, which serialize a bare `[Tensor]`
when you have no model record at all.

## Interop with PyTorch: pickle

For crossing the language border there is a second format. PyTorch
checkpoints are pickled dictionaries — `torch.save(model.state_dict(),
path)` — and `Torch.Serialize` speaks it directly through the
`IValue` type from the TorchScript bindings
([chapter 15](15-torchscript-and-jit.html)): `pickleSave` and
`pickleLoad` map dictionaries to `IVGenericDict`, tensors to
`IVTensor`, and so on. A state dict is built by hand from the
parameters and their names:

```haskell do
let Linear w b = model
    stateDict =
      IVGenericDict
        [ (IVString "weight", IVTensor (toDependent w)),
          (IVString "bias", IVTensor (toDependent b))
        ]
pickleSave stateDict "/tmp/state_dict.pth"
reloaded <- pickleLoad "/tmp/state_dict.pth"
```

```haskell eval
reloaded
```

The file we just wrote is a regular PyTorch checkpoint; on the Python
side it loads with plain `torch.load`:

```python
>>> torch.load("/tmp/state_dict.pth")
{'weight': tensor([[...]]), 'bias': tensor([...])}
```

The reverse direction works the same — save from Python with
`torch.save(dict(model.state_dict()), path)` (the `dict(...)` matters:
an `OrderedDict` subclass confuses the unpickler), then `pickleLoad`
it and match on the `IVGenericDict` to pull tensors out by name.
Names and shapes are yours to reconcile with your Haskell record;
for shipping a *whole* model rather than weights — architecture
included — TorchScript's `torch.jit.trace` plus
`Torch.Script.loadScript` is the better vehicle, as chapter 15 shows.

## The typed API

`Torch.Typed.Serialize` mirrors the untyped pair: `saveParameters`
and `loadParameters` work on any typed model whose parameters
flatten to an `HList` of tensors, and `loadParametersWithSpec`
builds the model from its spec and the file in one step. The shapes
in the model's type must match what was saved — the file format
cannot check this for you, but the surrounding program is checked as
always, so a shape mismatch surfaces at the load site rather than
mid-training.

One honest caveat applies to every format here: loading a checkpoint
executes no code, but pickle files from untrusted sources are still
untrusted input to a C++ parser — treat model files with the same
care as any other binary you download.
