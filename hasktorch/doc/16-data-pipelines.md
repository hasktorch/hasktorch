---
title: Data Pipelines
---

# Data Pipelines

Every training loop so far generated its batches on the fly. Real
datasets live in files, are bigger than memory, and need to be
decoded, shuffled, and batched while the GPU is busy with the
previous batch. `Torch.Data` is Hasktorch's answer: streaming data
loaders built on [pipes](https://hackage.haskell.org/package/pipes),
which read in constant memory and prefetch on background threads.

```haskell top hide
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
import Inliterate.Import (AskInliterate)
```

```haskell top
import Control.Monad (foldM)
import Control.Monad.Cont (runContT)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Pipes (ListT, enumerate)
import qualified Pipes.Prelude as P
import Pipes.Safe (SafeT, runSafeT)
import Torch
import Torch.Data.CsvDatastream
import Torch.Data.StreamedPipeline (datastreamOpts, streamFrom')
```

```haskell top hide
instance AskInliterate Tensor
instance AskInliterate Point
```

## Two kinds of dataset

`Torch.Data` mirrors the two dataset notions of PyTorch. A
*map-style* dataset is indexed — it knows its keys and can fetch any
sample directly (PyTorch's `Dataset`):

```haskell
class Ord k => Dataset m dataset k sample where
  getItem :: dataset -> k -> m sample
  keys :: dataset -> Set k
```

A *datastream* is only a stream — samples arrive in whatever order
the source produces them, which is all you can ask of a file being
read front to back, a network socket, or an infinite generator
(PyTorch's `IterableDataset`):

```haskell
class Monad m => Datastream m seed dataset sample where
  streamSamples :: dataset -> seed -> ListT m sample
```

Both are consumed the same way: a `streamFrom` function turns the
dataset into a `ListT` of samples — an ordinary pipes stream — and
the training loop is a fold over it. Nothing dataset-specific leaks
into the loop.

## CSV files decode into records

The most common concrete datastream is `CsvDatastream` from
`Torch.Data.CsvDatastream`. It decodes rows with
[cassava](https://hackage.haskell.org/package/cassava), so a row
schema is an ordinary Haskell record — the same move the named-tensor
chapter made for dimensions, applied to files:

```haskell top
data Point = Point
  { px :: Float,
    py :: Float
  }
  deriving (Generic, Show, FromRecord)
```

Let's make ourselves a small CSV file with points on the line
\(y = 3x + 1\):

```haskell do
writeFile "/tmp/points.csv" $
  unlines [show x ++ "," ++ show (3 * x + 1) | x <- [0.0, 0.01 .. 2.0 :: Float]]
```

A datastream over it is a value describing *how* to read the file —
nothing is opened yet:

```haskell top
stream :: CsvDatastream Point
stream = (csvDatastream "/tmp/points.csv") {batchSize = 16}
```

`streamFrom'` runs it. The stream lives in `SafeT IO` (from
`pipes-safe`) so the file handle is released even if the consumer
dies, and the result comes as a continuation (`runContT`) that scopes
the background prefetch thread. The stream yields batches — here
`V.Vector Point` of length 16:

```haskell do
batches <- runSafeT $
  runContT (streamFrom' datastreamOpts stream [()]) $ \input ->
    P.toListM (enumerate input)
```

```haskell eval
length batches
```

```haskell eval
V.toList (V.take 3 (head batches))
```

## Training is a fold over the stream

An epoch folds a model over the batches; `Pipes.Prelude.foldM` is the
loop. Batches become tensors with plain list conversions:

```haskell top
toBatch :: V.Vector Point -> (Tensor, Tensor)
toBatch b = (asTensor (map ((: []) . px) l), asTensor (map py l))
  where
    l = V.toList b

trainEpoch :: Linear -> ListT (SafeT IO) (V.Vector Point) -> SafeT IO Linear
trainEpoch model input = P.foldM step (pure model) pure (enumerate input)
  where
    step m batch = do
      let (xs, ys) = toBatch batch
          loss = mseLoss ys (squeezeAll (linear m xs))
      fst <$> liftIO (runStep m GD loss 1e-1)
```

Epochs re-run the stream from the top — the datastream value is
reusable, so an epoch loop is just `foldM` over epoch numbers:

```haskell do
model0 <- sample (LinearSpec 1 1)
trained <- runSafeT $
  foldM
    (\m _epoch -> runContT (streamFrom' datastreamOpts stream [()]) (trainEpoch m))
    model0
    [1 .. 20 :: Int]
```

The file said \(y = 3x + 1\); the model should agree (`Torch.NN`
uses duplicate record fields across its layer types, so we match the
`Linear` constructor positionally instead of using the `weight` and
`bias` accessors):

```haskell do
let Linear w b = trained
```

```haskell eval
toDependent w
```

```haskell eval
toDependent b
```

## Shuffling, prefetch, and parallelism

Everything above streamed sequentially. The knobs:

- **Shuffling.** `stream {bufferedShuffle = Just 1000}` keeps a
  buffer of 1000 records, yielding random elements from it as it
  refills — the streaming compromise, since a true shuffle would need
  the whole file. (Our file is sorted by `x`, so for real training
  you would want this; a buffer at least the file's size gives a full
  shuffle.) Map-style datasets can do better: `streamFromMap` takes a
  random generator and visits *keys* in shuffled order.
- **Prefetch.** `datastreamOpts` has a `bufferSize` (default 4):
  batches are produced on a background thread into a bounded queue,
  so decoding the next batch overlaps with training on the current
  one.
- **Sharding.** The `[()]` we passed is the list of *seeds*. Each
  seed starts its own concurrent copy of the stream, and the streams
  are interleaved — seeds can be file shards, worker ids, or random
  generators. For CSV there is one file, so one unit seed.
- **Parallel preprocessing.** `Torch.Data.Utils.pmap` maps a
  function over a stream on its own thread (the iris example uses it
  to move record-to-tensor conversion off the training thread), and
  `collate` regroups sample streams into batches.

`dropLast` (default `True`) discards a final ragged batch, which is
why 201 rows became 12 batches of 16 rather than 12 and a half.

## Where to look next

The [iris-classification](https://github.com/hasktorch/hasktorch/tree/master/examples/iris-classification)
example is this chapter on a real file; the
[static-mnist](https://github.com/hasktorch/hasktorch/tree/master/examples/static-mnist)
examples stream MNIST through the same interface with the typed API,
using the loader from `Torch.Vision`. The next chapter looks at what
the fold does with each batch: the [optimizers](17-optimizers.html).
