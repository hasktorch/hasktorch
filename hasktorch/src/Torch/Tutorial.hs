{-# OPTIONS_GHC -fno-warn-unused-imports #-}

{- | Hasktorch is a library for scientific computing and differentiable
programming.
-}
module Torch.Tutorial (
    -- $tutorial
) where

import Torch.Internal.Managed.Type.Context (manual_seed_L)

{- $setup
>>> manual_seed_L 123
>>> :set -XNoOverloadedLists
-}

{- $tutorial
= What is Hasktorch?
#introduction#

Hasktorch is a Haskell library for scientific computing and
differentiable programming.  It leverages @libtorch@ (the backend
library powering PyTorch) for efficient tensor manipulation and
automatic differentiation, while bringing to bear Haskell's expressive
type system and first-class support for for the functional programming
paradigm.

== Goal of this tutorial

The sequence of topics and examples here is loosely based on the
PyTorch tutorial [Deep Learning with PyTorch: A 60 Minute
Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
by Soumith Chintala.

In this tutorial we will implement a simple machine learning
model. Along the way, you will learn to

- create and manipulate tensors

- build computation graphs from tensors and compute gradients

- optimize parameters with respect to an objective function

= Usage
#usage#

The reader is encouraged to follow along with the examples in a GHCi session.

To start, import 'Torch':

>>> import Torch

== Tensors
#tensors#

A `Tensor` in Hasktorch is multidimensional array with a fixed shape
and element type.

For example, we can initialize a tensor with shape @[3, 4]@ and filled
with zeros using

>>> Torch.zeros' [3, 4]
Tensor Float [3,4] [[ 0.0000,  0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  0.0000]]

We can also initialize a tensor from a Haskell list using
'Torch.Tensor.asTensor':

>>> asTensor ([[4, 3], [2, 1]] :: [[Float]])
Tensor Float [2,2] [[ 4.0000   ,  3.0000   ],
                    [ 2.0000   ,  1.0000   ]]

Note that the numerical type of the tensor is inferred from the types of
the values in the list.

Scalar values are represented in Hasktorch as tensors with shape @[]@:

>>> asTensor 3.5
Tensor Double []  3.5000

We can get the scalar value back out using 'Torch.Tensor.asValue':

>>> asValue (asTensor 3.5)
3.5

=== Specifying Tensor parameters

In the previous section we initialized a tensor filled with zeros
using @zeros'@ (note the prime suffix). Hasktorch functions use a
convention where default versions of functions use a prime suffix. The
unprimed versions of these functions expect an additional parameter
specifying tensor parameters. For example:

@
  zeros :: [Int] -> 'Torch.TensorOptions' -> Tensor
@

@TensorOptions@ are typically specified by starting with
'Torch.TensorOptions.defaultOpts' and modifying using one or more of
the following:

- 'Torch.TensorOptions.withDType' configures the data type of the elements
- 'Torch.TensorOptions.withDevice' configures on which device the tensor is to be used
- others (see 'Torch.TensorOptions')

For example, to construct a matrix filled with zeros of dtype @Int64@:

>>> zeros [4, 4] (withDType Int64 defaultOpts)
Tensor Int64 [4,4] [[ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0]]

=== Tensor factories

Hasktorch comes with many "factory" functions similar to @zeros@ and
@zeros'@ useful for initializing common kinds of tensors. For example,
'Torch.TensorFactories.ones',
'Torch.TensorFactories.full',
'Torch.TensorFactories.eye',
and the primed versions of these. See 'Torch.TensorFactories' for a
complete list.

One useful class of factory functions are those suffixed with "-like"
(e.g. 'Torch.TensorFactories.onesLike'), which initialize a tensor
with the same dimensions as their argument. For example:

>>> let x = zeros' [3, 2]
>>> onesLike x
Tensor Float [3,2] [[ 1.0000   ,  1.0000   ],
                    [ 1.0000   ,  1.0000   ],
                    [ 1.0000   ,  1.0000   ]]

== Operations
#operations#

Most operations are pure functions, similar to Haskell standard library
math operations.

Tensors implement the @Num@ typeclass:

>>> let x = ones' [4]
>>> x + x
Tensor Float [4] [ 2.0000   ,  2.0000   ,  2.0000   ,  2.0000   ]

Some operations transform a tensor:

>>> Torch.relu (asTensor ([-1.0, -0.5, 0.5, 1] :: [Float]))
Tensor Float [4] [ 0.0000,  0.0000,  0.5000   ,  1.0000   ]

'Torch.Tensor.select' slices out a selection by specifying a dimension and index:

>>> let x = asTensor [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
>>> shape x
[4,1,3]

>>> select 2 1 x
Tensor Double [4,1] [[ 2.0000   ],
                     [ 5.0000   ],
                     [ 8.0000   ],
                     [ 11.0000   ]]

>>> let y = asTensor [1, 2, 3]
>>> Torch.select 0 1 y
Tensor Double []  2.0000

Values can be extracted from a tensor using @asValue@ so long as the
dtype matches the Haskell type:

>>> let x = asTensor ([2] :: [Int])
>>> let y = asValue x :: Int
>>> y
2

== Randomness

Create a randomly initialized matrix:

>>> x <-randIO' [2, 2]
>>> x
Tensor Float [2,2] [[ 0.2961   ,  0.5166   ],
                    [ 0.2517   ,  0.6886   ]]

Note that since random initialization returns a different result each
time, unlike other tensor constructors, is monadic reflecting the
context of an underlying random number generator (RNG) changing state.

Hasktorch includes variations of random tensor initializers in which the
RNG object is threaded explicitly rather than implicitly. See the
@Torch.Random@ module functions for details and variations. Samplers for
which the RNG is not explicit such as @randIO\'@ example above use the
@-IO@ suffix.


== Automatic Differentiation
#automatic-differentiation#

Automatic differentiation is achieved through the use of two primary
functions in the 'Torch.Autograd' module,
'Torch.Autograd.makeIndependent' and 'Torch.Autograd.grad'.

=== Independent Tensors
#independent-tensors#

@makeIndependent@ is used to instantiate an independent tensor variable
from which a compute graph is constructed for differentiation, while
@grad@ uses compute graph to compute gradients.

@makeIndependent@ takes a tensor as input and returns an IO action
which produces an 'Torch.Autograd.IndependentTensor':

>  makeIndependent :: Tensor -> IO IndependentTensor

What is the definition of the @IndependentTensor@ type produced by the
@makeIndependent@ action? It’s defined in the Hasktorch library as:

>  newtype IndependentTensor = IndependentTensor { toDependent :: Tensor }
>  deriving (Show)

Thus @IndependentTensor@ is simply a wrapper around the underlying
Tensor that is passed in as the argument to @makeIndependent@. Building
up computations using ops applied to the @toDependent@ tensor of an
@IndependentTensor@ will implicitly construct a compute graph to which
@grad@ can be applied.

All tensors have an underlying property that can be retrieved using
the 'Torch.Autograd.requiresGrad' function which indicates whether
they are a differentiable value in a compute graph. <#notes [1]>

>>> let x = asTensor [1, 2, 3]
>>> y <- makeIndependent (asTensor [4, 5, 6])
>>> let y' = toDependent y
>>> let z = x + y'
>>> requiresGrad x
False

>>> requiresGrad y'
True

>>> requiresGrad z
True

In summary, tensors that are computations of values derived from tensor
constructors (e.g. @ones@, @zeros@, @fill@, @randIO@ etc.) outside the
context of a @IndependentTensor@ are not differentiable. Tensors that
are derived from computations on the @toDependent@ value of an
@IndependentTensor@ are differentiable, as the above example
illustrates.

=== Gradients
#gradients#

Once a computation graph is constructed by applying ops and computing
derived quantities stemming from a @toDependent@ value of an
@IndependentTensor@, a gradient can be taken by using the @grad@
function specifying in the first argument tensor corresponding to
function value of interest and a list of @Independent@ tensor variables
that the the derivative is taken with respect to:

>  grad :: Tensor -> [IndependentTensor] -> [Tensor]

Let’s demonstrate this with a concrete example. We create a tensor and
derive an @IndependentTensor@ from it:

>>> x <- makeIndependent (ones' [2, 2])
>>> let x' = toDependent x
>>> x'
Tensor Float [2,2] [[ 1.0000   ,  1.0000   ],
                    [ 1.0000   ,  1.0000   ]]

Now do some computations on the dependent tensor:

>>> let y = x' + 2
>>> y
Tensor Float [2,2] [[ 3.0000   ,  3.0000   ],
                    [ 3.0000   ,  3.0000   ]]

Since y is dependent on the x independent tensor, it is differentiable:

>>> requiresGrad y
True

Applying more operations:

>>> let z = y * y * 3
>>> let out = mean z
>>> z
Tensor Float [2,2] [[ 27.0000   ,  27.0000   ],
                    [ 27.0000   ,  27.0000   ]]

Now retrieve the gradient:

>>> grad out [x]
[Tensor Float [2,2] [[ 4.5000   ,  4.5000   ],
                    [ 4.5000   ,  4.5000   ]]]

== Differentiable Programs (Neural Networks)
#differentiable-programs-neural-networks#

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
algebraic data type (ADT). The ADT can take on any shape that’s needed
to model the domain of interest, allowing a great deal of flexibility
and enabling all of Haskell’s strenghts in data modeling - can use sum
or product types, nest types, etc. The ADT can implement various
typeclasses to take on other functionality.

The core interface that defines capability specific to differentiable
programming is the 'Torch.NN.Parameterized' typeclass:

>  class Parameterized f where
>    flattenParameters :: f -> [Parameter]
>    default flattenParameters :: (Generic f, Parameterized' (Rep f)) => f -> [Parameter]
>    flattenParameters f = flattenParameters' (from f)
>
>    replaceOwnParameters :: f -> ParamStream f
>    default replaceOwnParameters :: (Generic f, Parameterized' (Rep f)) => f -> ParamStream f
>    replaceOwnParameters f = to <$> replaceOwnParameters' (from f)

Note @Parameter@ is simply a type alias for @IndependentTensor@ in the
context of neural networks (i.e. @type Parameter = IndependentTensor@).

The role of @flattenParameters@ is to unroll any arbitrary ADT
representation of a neural network into a standard flattened
representation consisting a list of @IndependentTensor@ which is used to
compute gradients.

@replaceOwnParameters@ is used to update parameters. ParamStream is a
type alias for a State type with state represented by a @Parameter@ list
and a value parameter corresponding to the ADT defining the model.

>  type ParamStream a = State [Parameter] a

Note the use of generics. Generics allow the compiler to usually
automatically derive @flattenParameters@ and @replaceOwnParameter@
instances without any code if your type is built up on tensors,
containers of tensors, or other types that are built from tensor values
(for example, layer modules provided in @Torch.NN@. In many cases, as
you’ll see in the following examples, you will only need to add

>  instance Parameterized MyNeuralNetwork

(where @MyNeuralNetwork@ is an ADT definition for your model) and the
compiler will derive implementations for the @flattenParameters@ and
@replaceOwnParameters@.

=== Linear Regression
#linear-regression#

Lets start with a simple example of linear regression. Here we generate
random data with an underlying affine relationship between the inputs
and outputs, then fit a linear regression to reproduce that
relationship.

This example is adapted from
<https://github.com/hasktorch/hasktorch/tree/master/examples/regression>.

In a standard supervised learning model, the neural network is
initialized using a randomized initialization scheme. An iterative
optimization is performed such that at each iteration a batch.

>  module Main where
>
>  import Control.Monad (when)
>  import Torch
>
>  groundTruth :: Tensor -> Tensor
>  groundTruth t = squeezeAll $ matmul t weight + bias
>    where
>      weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
>      bias = full' [1] (3.14 :: Float)
>
>  model :: Linear -> Tensor -> Tensor
>  model state input = squeezeAll $ linear state input
>
>  main :: IO ()
>  main = do
>      init <- sample $ LinearSpec{in_features = numFeatures, out_features = 1}
>      randGen <- mkGenerator (Device CPU 0) 12345
>      (trained, _) <- foldLoop (init, randGen) 2000 $ \(state, randGen) i -> do
>          let (input, randGen') = randn' [batchSize, numFeatures] randGen
>              (y, y') = (groundTruth input, model state input)
>              loss = mseLoss y y'
>          when (i `mod` 100 == 0) $ do
>              putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
>          (newParam, _) <- runStep state GD loss 5e-3
>          pure (replaceParameters state newParam, randGen')
>      pure ()
>    where
>      batchSize = 4
>      numFeatures = 3

Note the expression of the architecture in the 'Torch.NN.linear'
function (a single linear layer, or alternatively a neural network
with zero hidden layers), does not require an explicit representation
of the compute graph, but is simply a composition of tensor
ops. Because of the autodiff mechanism described in the previous
section, the graph is constructed automatically as pure functional ops
are applied, given a context of a set of independent variables.

The @init@ variable is initialized as a @Linear@ type (defined in
@Torch.NN@) using @sample@ which randomly initializes a @Linear@ value.

@Linear@ is a built-in ADT implementing the @Parameterized@ typeclass
and representing a fully connected linear layer, equivalent to linear
regression when no hidden layers are present.

@init@ is passed into the 'Torch.Optim.foldLoop'@<#notes [2]> as the @state@
variable.

A new list of @Parameter@ values is passed back from
'Torch.Optim.runStep' (which calls @grad@ to retrieve gradients, given
a loss function, learning rate, and optimizer) and the typeclass
function @replaceParameters@ is used to update the model at each
iteration.

Initialization is discussed in more detail in the following section

=== Weight Initialization
#weight-initialization#

Random initialization of weights is not a pure function since two
random initializations return different values. Initialization occurs
by calling the 'Torch.NN.sample' function for an ADT (@spec@)
implementing the 'Torch.NN.Randomizable' typeclass:

>  class Randomizable spec f | spec -> f where
>    sample :: spec -> IO f

In a typical (but not required) usage, @f@ is an ADT that implements the
@Parameterized@ typeclass, so that there’s a pair of types - a
specification type implementing the @spec@ input to @sample@ and a type
implementing @Parameterizable@ representing the model state.

For example, a linear fully connected layer is provided by the
@Torch.NN@ module and defined therein as:

>  data Linear = Linear { weight :: Parameter, bias :: Parameter } deriving (Show, Generic)

and is typically used with a specification type:

>  data LinearSpec = LinearSpec { in_features :: Int, out_features :: Int }
>    deriving (Show, Eq)

Putting this together, in untyped tensor usage, the user can implement
custom models or layers implementing the @Parameterizable@ typeclass
built up from other ADTs implementing @Parameterizable@. The shape of
the data required for initialization is described by a type implementing
@Randomizable@’s @spec@ parameter, and the @sample@ implementation
specifies the default weight initialization.

Note this initialization approach is specific to untyped tensors. One
consequence of using typed tensors is that the information in these
@spec@ types is reflected in the type itself and thus are not needed.

What if you want to use a custom initialization that differs from the
default? You can define an alternative function with the same signature
@spec -> IO f@ and use the alternative function instead of @sample@.

=== Optimizers
#optimizers#

Optimization implementations are functions that take as input the
current parameter values of a model, parameter gradient estimates of the
loss function at those parameters for a single batch, and a
characteristic learning describing how large a perturbation to make to
the parameters in order to reduce the loss. Given those inputs, they
output a new set of parameters.

In the simple case of stochastic gradient descent, the function to
output a new set of parameters is to subtract from the current parameter
\(\theta\), the gradient of the loss \(\nabla J\) scaled by the learning
rate \(\eta\):

\[\theta_{i+1} = \theta_i - \eta \nabla J(\theta)\]

While stochastic gradient descent is a stateless function of the
parameters, loss, and gradient, some optimizers have a notion of
internal state that is propagated from one step to the step, for
example, retaining and updating momentum between steps:

\[\begin{gathered}
    \Delta \theta_i = \alpha \Delta \theta_{i-1} - \eta \nabla J(\theta) \\
    \theta_{i+1} = \theta_i + \Delta \theta_i
\end{gathered}\]

In this case, the momentum term \(\Delta \theta_i\) is carried forward
as internal state of the optimizer that is propagated to the next step.
\(\alpha\) is an optimizer parameter which determines a weighting on the
momentum term relative to the gradient.

Implementation of an optimizer consists of defining an ADT describing
the optimizer state and a @step@ function that implements a single step
perturbation given the learning rate, loss gradients, current
parameters, and optimizer state.

This function interface is described in the 'Torch.Optim.Optimizer'
typeclass interface:

>  class Optimizer o where
>      step :: LearningRate -> Gradients -> [Tensor] -> o -> ([Tensor], o)

@Gradients@ is a newtype wrapper around a list of tensors to make
intent explicit: @newtype Gradients = Gradients [Tensor]@.

Hasktorch provides built-in optimizer implementations in @Torch.Optim@.
Some illustrative example implementations follow.

Being stateless, stochastic gradient descent has an ADT that has only
one constructor value:

>  data GD = GD

and implements the step function as:

>  instance Optimizer GD where
>      step lr gradients depParameters dummy = (gd lr gradients depParameters, dummy)
>          where
>          step p dp = p - (lr * dp)
>          gd lr (Gradients gradients) parameters = zipWith step parameters gradients

The use of an optimizer was illustrated in the linear regression example
using the function @runStep@

>  (newParam, _) <- runStep state GD loss 5e-3

In this case the new optimizer state returned is ignored (as @_@) since
gradient descent does not have any internal state. Under the hood,
@runStep@ does a little bookkeeping making independent variables from a
model, computing gradients, and passing values to the @step@ function.
Usually a user can ignore the details and just pass model parameters and
the optimizer to runStep as an abstracted interface which takes
parameter values, the optimizer value, loss (a tensor), and learning
rate as input and returns new parameters and an updated optimizer value.

>  runStep :: (Parameterized p, Optimizer o) =>
>          p -> o -> Tensor -> LearningRate -> IO ([Parameter], o)

= Typed Tensors
#typed-tensors#

Typed tensors provide an alternative API for which tensor
characteristics are encoded in the type of the tensor. The Hasktorch
library is layered such that [ TODO ]

Using typed tensors increases the expressiveness of program invariants
that can be automatically checked by GHC at the cost of needing to be
more explicit in writing code and also requiring working with Haskell’s
type-level machinery. Type-level Haskell programming has arisen through
progressive compiler iteration leading to mechanisms that have a higher
degree of complexity compared to value-level Haskell code or other
languages such as Idris which were designed with type-level computations
from inception. In spite of these compromises, Haskell offers enough
capability to express powerful type-level representations of models that
is matched by few other languages used for production applications.

Things we can do in typed Hasktorch:

-   specify, check, and infer tensor shapes at compile time

-   specify, check, and infer tensor data types at compile time

-   specify, check, and infer tensor compute devices at compile time

We can encode all Hasktorch tensor shapes on the type level using
type-level lists and natural numbers:

>  type EmptyShape = '[]
>  type OneDimensionalShape (a :: Nat) = '[a]
>  type TwoDimensionalShape (a :: Nat) (b :: Nat) = '[a, b]
>  ...

Tensor data types and compute device types are lifted to the type level
using the @DataKinds@ language extension:

>  type BooleanCPUTensor (shape :: [Nat]) = Tensor '(CPU,  0) 'Bool  shape
>  type IntCUDATensor    (shape :: [Nat]) = Tensor '(CUDA, 1) 'Int64 shape

Devices are represented as tuples consisting of a @DeviceType@ (here
@CPU@ for the CPU and @CUDA@ for a CUDA device, respectively) and a
device id (here the @Nat@s @0@ and @1@, respectively).

It is a common misconception that specifying tensor properties at
compile time is only possible if all tensor properties are constants and
are statically known. If this were the case, then we could only write
functions over fully specified tensors, say,

>  boring :: BooleanCPUTensor '[] -> BooleanCPUTensor '[]
>  boring = id

Fortunately, Haskell has the ability to reason about type variables.
This feature is called parametric polymorphism. Consider this simple
example of a function:

>  tensorNoOp
>    :: forall (shape :: [Nat]) (dtype :: DType) (device :: (DeviceType, Nat))
>     . Tensor device dtype shape
>    -> Tensor device dtype shape
>  tensorNoOp = id

Here, @shape@, @dtype@, and @device@ are type variables that have been
constrained to be of kind shape, data type, and device, respectively.
The universal quantifier @forall@ implies that this function is
well-defined for all inhabitants of the types that are compatible with
the type variables and their constraints.

The @tensorNoOp@ function may seem trivial, and that is because its type
is very strongly constrained: Given any typed tensor, it must return a
tensor of the same shape and with the same data type and on the same
device. Besides by means of the identity, @id@, there are not many ways
in which this function can be implemented.

There is a connection between a type signature of a function and
mathematical proofs. We can say that the fact that a function exists is
witnessed by its implementation. The implementation, @tensorNoOp = id@,
is the proof of the theorem stated by @tensorNoOp@’s type signature. And
here we have a proof that we can run this function on any device and for
any data type and tensor shape.

This is the essence of typed Hasktorch. As soon as the compiler gives
its OK, we hold a proof that our program will run without shape or CUDA
errors.

#notes#

1.  PyTorch users will be familiar with this as the @requires_grad@
    member variable for the PyTorch tensor type. The Hasktorch mechanism
    is distinct from PyTorch’s mechanism - by only allowing gradients to
    be applied in the context of a set of @IndependentTensor@ variables,
    it allows ops to be semantically pure and preserve referential
    transparency.

2.  @foldLoop@ is a convenience function defined in terms of @foldM@ as
    @foldLoop x count block = foldM block x ([1 .. count] :: [a])@
-}
