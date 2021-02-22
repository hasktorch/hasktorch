---
title: Differentiable Programs
---

# Differentiable Programs (Neural Networks)

From a functional programming perspective, a neural network is
represented by data and functions, much like any other functional
program. The only distinction that differentiates neural networks from
any other functional program is that it implements a small interface
surface to support differentiation. Thus, we can consider neural
networks to be \"differentiable functional programming\".

The data in neural networks are the values to be fitted that
parameterize the functions which carry out the inference operation and
are modified based on gradients of through those functions.

As with a regular Haskell program, this data is represented by an
algebraic data type (ADT). The ADT can take on any shape that's needed
to model the domain of interest, allowing a great deal of flexibility
and enabling all of Haskell's strenghts in data modeling - can use sum
or product types, nest types, etc. The ADT can implement various
typeclasses to take on other functionality.

The core interface that defines capability specific to differentiable
programming is the `Torch.NN.Parameterized` typeclass:

```haskell
class Parameterized f where
  flattenParameters :: f -> [Parameter]
  default flattenParameters :: (Generic f, Parameterized' (Rep f)) => f -> [Parameter]
  flattenParameters f = flattenParameters' (from f)

  replaceOwnParameters :: f -> ParamStream f
  default replaceOwnParameters :: (Generic f, Parameterized' (Rep f)) => f -> ParamStream f
  replaceOwnParameters f = to <$> replaceOwnParameters' (from f)
```

Note `Parameter` is simply a type alias for `IndependentTensor` in the
context of neural networks (i.e. `type Parameter =
IndependentTensor`).

The role of `flattenParameters` is to unroll any arbitrary ADT
representation of a neural network into a standard flattened
representation consisting a list of `IndependentTensor` which is used
to compute gradients.

`replaceOwnParameters` is used to update parameters. ParamStream is a
type alias for a State type with state represented by a `Parameter`
list and a value parameter corresponding to the ADT defining the
model.

```haskell
type ParamStream a = State [Parameter] a
```

Note the use of generics. Generics allow the compiler to usually
automatically derive `flattenParameters` and `replaceOwnParameter`
instances without any code if your type is built up on tensors,
containers of tensors, or other types that are built from tensor
values (for example, layer modules provided in `Torch.NN`. In many
cases, as you'll see in the following examples, you will only need to
add

```haskell
instance Parameterized MyNeuralNetwork
```

(where `MyNeuralNetwork` is an ADT definition for your model) and the
compiler will derive implementations for the `flattenParameters` and
`replaceOwnParameters`.
