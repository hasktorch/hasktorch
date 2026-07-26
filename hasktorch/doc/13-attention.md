---
title: Attention, Equation by Equation
---

# Attention, Equation by Equation

The library ships a full transformer
(`Torch.Typed.NN.Transformer`, ~640 lines), and it is fair to say
that reading it does not make the *idea* of attention visible. This
chapter makes the opposite trade: a complete, trainable decoder block
where each equation of the paper is one definition, with the paper's
shape annotations as checked types. The full listing is
`test/Torch/Typed/AttentionSpec.hs`; CI trains it and checks that it
learns.

## The three equations

A paper writes attention as

> M ∈ ℝ^{s×s},  M_{ij} = 0 if j ≤ i, −∞ otherwise
>
> Attention(Q, K, V) = softmax(QKᵀ/√d + M) V

and the code says exactly that. The mask is a tensor *defined by its
formula*, through `tabulate`:

```haskell
causalMask :: T '[S, S]
causalMask = toUnnamed (tabulateList @'[Vector S, Vector S] @'D.Float @Dev mask)
  where
    mask [i, j] = if j <= i then 0 else -1 / 0
```

and attention is its equation, with the ℝ^{s×e} annotations moved
into the type where the compiler can read them:

```haskell
attend :: T '[S, E] -> T '[S, E] -> T '[S, E] -> T '[S, E]
attend q k v =
  softmax @1 (divScalar sqrtD (q `matmul` transpose2D k) + causalMask) `matmul` v
```

A decoder block is four more lines, each one a line of the
architecture diagram:

```haskell
gptForward GPT {..} tokens = forward unembed x''
  where
    x   = embedding @'Nothing False False (toDependent embed) tokens
    x'  = x  + forward wo (attend (forward wq xn) (forward wk xn) (forward wv xn))
    x'' = x' + forward ff2 (relu (forward ff1 (forward norm2 x')))
    xn  = forward norm1 x
```

The parameters are an ordinary record of `Linear`s, `LayerNorm`s and
an embedding table with `deriving (Generic, Parameterized)`; the
whole model, mask to logits, is about sixty lines.

## And it learns

The test trains this block with Adam on next-token prediction over a
repeating sequence and asserts the outcome: the loss starts near
ln 4 ≈ 1.39 (uniform guessing over four tokens), ends below 0.1
after a thousand steps, and the trained block predicts every next
token correctly. The whole run takes about a second on CPU. That
closes the loop this chapter is about: the code above is not
pseudocode — it is the model, the types are its shape discipline, and
the test is its meaning.

## Why the library version is ten times longer

`Torch.Typed.NN.Transformer` is not long because attention is
complicated. It is long because it is *general*: separate embedding
dimensions for queries, keys and values; any number of heads, with
the reshape/transpose plumbing that heads require; optional key
padding masks and attention masks; dropout at four sites; batch
dimensions; and polymorphism over dtype and device, each generalized
parameter carrying its constraints through every signature. The
mathematics you can see above is in there, spread thin across that
generality.

The practical reading order is therefore: understand this chapter's
sixty lines first, then read the library transformer as "the same
equations, made reusable" — and let the types tell you which of its
parameters is which, because every shape annotation in its long
signatures is one of the ℝ^{...} superscripts from the paper, checked.
