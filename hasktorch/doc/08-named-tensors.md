---
title: Named Tensors and Lenses
---

# Named Tensors and Lenses

Typed tensors (see [Typed Tensors](07-typed-tensors.html)) check the
*sizes* of dimensions at compile time. Named tensors go one step
further: they give dimensions *meanings*. A `Tensor` of shape
`'[2, 3]` and another of shape `'[2, 3]` are the same type even when
one holds RGB channels and the other holds YCoCg channels; a
`NamedTensor` distinguishes them.

## Shapes made of types

A `NamedTensor` is indexed by a list of *type constructors* rather
than a list of naturals:

```haskell
import Torch.Typed

data RGB a = RGB
  { r :: a,
    g :: a,
    b :: a
  }
  deriving (Show, Eq, Generic)

newtype Batch (n :: Nat) a = Batch (Vector n a) deriving (Generic)

type Image = NamedTensor '( 'D.CPU, 0) 'D.Float '[Batch 2, RGB]
```

The runtime shape is derived from the types by the `ToNat` type
family: `Batch 2` contributes 2, and `RGB` contributes 3 because it is
a record with three fields. No registration is needed — `ToNat` walks
the `Generic` representation, so any record or sized-vector newtype
works as a dimension out of the box.

`'[Batch 2, RGB]` and `'[Batch 2, YCoCg]` both erase to `[2, 3]` at
runtime, but they are different types, and converting between them is
an explicit, checked function. This is the property that plain size
types (including those of array languages like Futhark) cannot
express.

## Field lenses

Because `RGB` is a record, its fields address positions of the
dimension. The `field` lens extracts one:

```haskell
import Torch.Typed.Lens

red :: NamedTensor device dtype '[Batch 2] -- the RGB dimension is dropped
red = image ^. field @"r"
```

Field names are checked: `field @"q"` on a shape containing `RGB`
does not compile. Whole dimensions can be addressed by name with the
`name` traversal:

```haskell
channels :: Traversal' (NamedTensor dev dt '[Batch 2, RGB])
                       (NamedTensor dev dt '[Batch 2])
channels = name @RGB
```

A worked conversion, from the test suite:

```haskell
toYCoCG :: NamedTensor device dtype '[Vector n, RGB]
        -> NamedTensor device dtype '[Vector n, YCoCg]
toYCoCG rgb =
  set (field @"y")  ((r + g * 2 + b) / 4) $
  set (field @"co") ((r - b) / 2) $
  set (field @"cg") ((- r + g * 2 - b) / 4) $
  def
  where
    r = rgb ^. field @"r"
    g = rgb ^. field @"g"
    b = rgb ^. field @"b"
```

Nothing in this code mentions a numeric channel index.

## Tensors as functions of their index

`Torch.Typed.Representable` treats a named tensor as what it
mathematically is: a function from an index to an element.

```haskell
import Torch.Typed.Representable

-- Log (index type) of Image is HList '[Finite 2, Finite 3]:
-- one bounds-checked index per dimension.

image :: Image
image = tabulate (\(i :. j :. HNil) -> fromIntegral (fromEnum i) * 10
                                     + fromIntegral (fromEnum j))

x :: Float
x = index image (1 :. 2 :. HNil)
```

`tabulate` builds the whole tensor from the function in a single
batched call; `index` reads one element. The laws
`index (tabulate f) i == f i` and `tabulate (index t) == t` are
checked in the test suite against real tensors.

Note the granularity: `tabulate` is efficient (one `asTensor` call),
`index` costs a few FFI calls per element, so it is for spot reads,
not for loops over all elements. For whole-tensor element-wise
computation, see [Graded and Staged Tensor
Programs](11-graded-and-staged.html).

## Lenses over whole models: changing dtype and device

Field and dimension lenses address parts of one tensor. The other
direction lenses work in Hasktorch is *outward*: traversing every
tensor inside an arbitrary structure — a model record, a tuple of
parameters, a list of batches.

In the untyped API this is `Torch.Lens`'s `HasTypes` traversal, and
the conversions built on it:

```haskell
import Torch

modelHalf = toType Half model      -- every tensor inside, converted
modelCuda = toDevice (Device CUDA 0) model
edited    = over (types @Tensor @MyModel) f model  -- any tensor rewrite
```

`HasTypes` is derived generically, so any record of tensors (or of
records of tensors) works without instances. This is how you switch a
whole network to half precision, or move it to a GPU, in one line —
but nothing in `model`'s *type* records that it happened.

The typed API has the same one-liners, with one important difference:
the conversion changes the type of the structure.

```haskell
import qualified Torch.Typed.DType as D
import qualified Torch.Typed.Device as Dev

model                                :: Linear 10 1 'D.Float '( 'D.CPU, 0)
D.toDType @'D.Half @'D.Float model   :: Linear 10 1 'D.Half  '( 'D.CPU, 0)
Dev.toDevice @'( 'D.CUDA, 0) @'( 'D.CPU, 0) model
                                     :: Linear 10 1 'D.Float '( 'D.CUDA, 0)
```

`HasToDType`/`HasToDevice` are again derived generically for records
of layers, and the `ReplaceDType` type family rewrites the dtype
parameter everywhere it occurs in the model's type. The functional
dependencies make the conversion bidirectional and unambiguous.

The payoff is downstream: after `toDType @'D.Half`, the model *is* a
half-precision model as far as GHC is concerned. Every forward pass,
loss, and optimizer step is now checked against `'D.Half`, so a stray
`'D.Float` batch fed to it — the classic silent-upcast bug of mixed
precision work — is a compile-time error, not a performance mystery.

The `types` traversal itself also works at *typed* targets, where it
gains an ability the untyped version cannot have: shape selectivity.
`Torch.Typed.Lens` provides the leaf instances, and then

```haskell
over (types @(Tensor '( 'D.CPU, 0) 'D.Float '[2, 3])) f model
flattenValues (types @(Parameter '( 'D.CPU, 0) 'D.Float '[5, 10])) model
replaceValues (types @(Tensor ... '[2, 3])) model newTensors
```

visit exactly the tensors (or parameters) of the named device, dtype
and shape inside a structure — a `'[3, 4]` layer sitting next to a
`'[2, 3]` one is left untouched. Where untyped `types @Tensor` means
"every tensor", the typed version means "every tensor *of this
structure*", which is what makes targeted surgery on a model — swap
these embeddings, freeze that projection — expressible as one
traversal.

## Lineage: "Tensor Considered Harmful"

The case for named dimensions was made by Alexander Rush's essay
[Tensor Considered Harmful](https://blog.rush-nlp.com/named-tensor.html)
(2019, with the `namedtensor` library; it later grew into the *Named
Tensor Notation* paper and PyTorch's experimental named tensors). The
essay diagnoses three traps of positional tensors — dimensions kept
private *by convention* only, broadcasting *by alignment* rather than
by meaning, and access *by comments* (`# batch x height x width`) —
and proposes a discipline: dimensions get human-readable names, no
function takes a `dim` argument, broadcasting matches by name, and
positional indexing is banned.

Hasktorch's named tensors are that proposal pushed one level up: the
names live in the *types*, so the discipline is checked by the
compiler instead of at runtime.

| Tensor Considered Harmful | Hasktorch |
|---|---|
| names are runtime strings: `("batch", "height", …)` | names are types: `'[Batch n, Height, Width]` — misspelling one is a compile error, and each name carries its *size* |
| `.mean("batch")` — "no function should have a dim argument" | `meanNamedDim @Batch`, `sumNamedDim @Batch`, `sortNamedDim @Seq` — `FindDim` locates the axis by name at compile time |
| broadcasting by name matching, checked when the op runs | shapes are types, so an inconsistent combination fails to *compile*; underneath, execution still uses positional kernels |
| "ban dimension based indexing" | indexing is by one bounds-checked `Finite` per named dimension (`index t (b :. h :. HNil)`, chapter section above) |
| `.split(h=…)` / `.stack(bh=…)` reshape by name | `dimGroup` / `dimUngroup` name a compound axis (`Compose (Vector 3) RGB`), and `dimUp` / `dimDown` go further — the dimension becomes actual Haskell structure ([chapter 12](12-dimensions.html)) |
| "private dimensions should be protected": a rotation should not know about `batch` | `vmap` and `emapS` take functions written against the *element* shape only; the batch dimension does not appear in their types at all ([chapter 12](12-dimensions.html)) |
| names should impose no runtime cost | `NamedTensor` is a zero-cost wrapper over the positional tensor; every name is erased at compile time |

Two things here have no counterpart in the essay. Names are
*structured*: a dimension can be a record (`RGB`), so its positions
are fields you can pattern-match — the essay's names identify an
axis, but cannot give its positions meanings. And the static setting
changes what "checking" means: the essay's names are verified while
the program runs, per operation; here a shape mistake is a type
error before anything runs, which is also why the names can be
erased completely.
