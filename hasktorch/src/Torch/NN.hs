{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.NN where

import Control.Applicative (Applicative (liftA2))
import Control.Monad.State.Strict
import Data.Kind
import Data.Proxy
import Data.Type.Bool
import GHC.Generics
import GHC.TypeLits
import System.IO.Unsafe (unsafePerformIO)
import Torch.Autograd
import Torch.Functional
import Torch.Initializers
import Torch.Internal.Cast (cast3)
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import Torch.Tensor
import Torch.TensorFactories (ones', randIO', randnIO')

type Parameter = IndependentTensor

type ParamStream a = State [Parameter] a

nextParameter :: ParamStream Parameter
nextParameter = do
  params <- get
  case params of
    [] -> error "Not enough parameters supplied to replaceParameters"
    (p : t) -> do put t; return p

class HasForward model input where
  type Output model input :: Type
  type Output model input = GOutput (Rep model) (Rep input)
  forward :: model -> input -> Output model input

class GHasForward (model :: Type -> Type) (input :: Type -> Type) where
  type GOutput model input :: Type
  gForward :: forall a b. model a -> input b -> GOutput model input

instance GHasForward U1 U1 where
  type GOutput U1 U1 = U1 ()
  gForward U1 U1 = U1

data ModelRandomness = Deterministic | Stochastic

-- TODO: remove placeholder random state 'G', replace with (typed version of):
-- https://github.com/hasktorch/hasktorch/blob/35e447da733c3430cd4a181c0e1d1b029b68e942/hasktorch/src/Torch/Random.hs#L38
data G

-- TODO: move to typelevel utils (maybe Torch.Typed.Aux?)
type family Contains (f :: k) (a :: Type) :: Bool where
  Contains a a = 'True
  Contains (f g) a = Contains f a || Contains g a
  Contains _ _ = 'False

type family ModelRandomnessR (output :: Type) :: (ModelRandomness, Type) where
  ModelRandomnessR (G -> (output, G)) =
    If
      (Contains output G)
      (TypeError (Text "The random generator appears in a wrong position in the output type."))
      '( 'Stochastic, output)
  ModelRandomnessR output =
    If
      (Contains output G)
      (TypeError (Text "The random generator appears in a wrong position in the output type."))
      '( 'Deterministic, output)

class
  HasForwardProduct
    (modelARandomness :: ModelRandomness)
    modelA
    inputA
    outputA
    (modelBRandomness :: ModelRandomness)
    modelB
    inputB
    outputB
  where
  type OutputProduct modelARandomness modelA inputA outputA modelBRandomness modelB inputB outputB :: Type
  forwardProduct ::
    Proxy modelARandomness ->
    modelA ->
    inputA ->
    Proxy outputA ->
    Proxy modelBRandomness ->
    modelB ->
    inputB ->
    Proxy outputB ->
    OutputProduct modelARandomness modelA inputA outputA modelBRandomness modelB inputB outputB

class
  HasForwardSum
    (modelARandomness :: ModelRandomness)
    modelA
    inputA
    outputA
    (modelBRandomness :: ModelRandomness)
    modelB
    inputB
    outputB
  where
  type OutputSum modelARandomness modelA inputA outputA modelBRandomness modelB inputB outputB :: Type
  forwardSum ::
    Proxy modelARandomness ->
    Proxy modelBRandomness ->
    Either modelA modelB ->
    Either inputA inputB ->
    Proxy (Either outputA outputB) ->
    OutputSum modelARandomness modelA inputA outputA modelBRandomness modelB inputB outputB

--
-- Deterministic instances
--

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ outputA,
    HasForward modelB inputB,
    Output modelB inputB ~ outputB
  ) =>
  HasForwardProduct 'Deterministic modelA inputA outputA 'Deterministic modelB inputB outputB
  where
  type OutputProduct 'Deterministic modelA inputA outputA 'Deterministic modelB inputB outputB = (outputA, outputB)
  forwardProduct _ modelA inputA _ _ modelB inputB _ = (forward modelA inputA, forward modelB inputB)

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ outputA,
    HasForward modelB inputB,
    Output modelB inputB ~ outputB
  ) =>
  HasForwardSum 'Deterministic modelA inputA outputA 'Deterministic modelB inputB outputB
  where
  type OutputSum 'Deterministic modelA inputA outputA 'Deterministic modelB inputB outputB = Maybe (Either outputA outputB)
  forwardSum _ _ (Left modelA) (Left inputA) _ = Just . Left $ forward modelA inputA
  forwardSum _ _ (Right modelB) (Right inputB) _ = Just . Right $ forward modelB inputB
  forwardSum _ _ _ _ _ = Nothing

--
-- Stochastic mixed instances
--

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ (G -> (outputA, G)),
    HasForward modelB inputB,
    Output modelB inputB ~ outputB
  ) =>
  HasForwardProduct 'Stochastic modelA inputA outputA 'Deterministic modelB inputB outputB
  where
  type OutputProduct 'Stochastic modelA inputA outputA 'Deterministic modelB inputB outputB = G -> ((outputA, outputB), G)
  forwardProduct _ modelA inputA _ _ modelB inputB _ = \g -> let (outputA, g') = forward modelA inputA g in ((outputA, forward modelB inputB), g')

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ outputA,
    HasForward modelB inputB,
    Output modelB inputB ~ (G -> (outputB, G))
  ) =>
  HasForwardProduct 'Deterministic modelA inputA outputA 'Stochastic modelB inputB outputB
  where
  type OutputProduct 'Deterministic modelA inputA outputA 'Stochastic modelB inputB outputB = G -> ((outputA, outputB), G)
  forwardProduct _ modelA inputA _ _ modelB inputB _ = \g -> let (outputB, g') = forward modelB inputB g in ((forward modelA inputA, outputB), g')

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ (G -> (outputA, G)),
    HasForward modelB inputB,
    Output modelB inputB ~ outputB
  ) =>
  HasForwardSum 'Stochastic modelA inputA outputA 'Deterministic modelB inputB outputB
  where
  type OutputSum 'Stochastic modelA inputA outputA 'Deterministic modelB inputB outputB = G -> (Maybe (Either outputA outputB), G)
  forwardSum _ _ (Left modelA) (Left inputA) _ = \g -> let (outputA, g') = forward modelA inputA g in (Just $ Left outputA, g')
  forwardSum _ _ (Right modelB) (Right inputB) _ = \g -> (Just . Right $ forward modelB inputB, g)
  forwardSum _ _ _ _ _ = \g -> (Nothing, g)

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ outputA,
    HasForward modelB inputB,
    Output modelB inputB ~ (G -> (outputB, G))
  ) =>
  HasForwardSum 'Deterministic modelA inputA outputA 'Stochastic modelB inputB outputB
  where
  type OutputSum 'Deterministic modelA inputA outputA 'Stochastic modelB inputB outputB = G -> (Maybe (Either outputA outputB), G)
  forwardSum _ _ (Left modelA) (Left inputA) _ = \g -> (Just . Left $ forward modelA inputA, g)
  forwardSum _ _ (Right modelB) (Right inputB) _ = \g -> let (outputA, g') = forward modelB inputB g in (Just $ Right outputA, g')
  forwardSum _ _ _ _ _ = \g -> (Nothing, g)

--
-- Fully-stochastic instances
--

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ (G -> (outputA, G)),
    HasForward modelB inputB,
    Output modelB inputB ~ (G -> (outputB, G))
  ) =>
  HasForwardProduct 'Stochastic modelA inputA outputA 'Stochastic modelB inputB outputB
  where
  type OutputProduct 'Stochastic modelA inputA outputA 'Stochastic modelB inputB outputB = G -> ((outputA, outputB), G)
  forwardProduct _ modelA inputA _ _ modelB inputB _ = runState $ do
    outputA <- state (forward modelA inputA)
    outputB <- state (forward modelB inputB)
    return (outputA, outputB)

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ (G -> (outputA, G)),
    HasForward modelB inputB,
    Output modelB inputB ~ (G -> (outputB, G))
  ) =>
  HasForwardSum 'Stochastic modelA inputA outputA 'Stochastic modelB inputB outputB
  where
  type OutputSum 'Stochastic modelA inputA outputA 'Stochastic modelB inputB outputB = G -> (Maybe (Either outputA outputB), G)
  forwardSum _ _ (Left modelA) (Left inputA) _ = \g -> let (outputA, g') = forward modelA inputA g in (Just $ Left outputA, g')
  forwardSum _ _ (Right modelB) (Right inputB) _ = \g -> let (outputA, g') = forward modelB inputB g in (Just $ Right outputA, g')
  forwardSum _ _ _ _ _ = \g -> (Nothing, g)

-- TODO: move to Torch.Typed.Prelude?
type family Fst (t :: (k, k')) :: k where
  Fst '(x, _) = x

type family Snd (t :: (k, k')) :: k' where
  Snd '(_, y) = y

instance
  ( '(modelARandomness, outputA) ~ ModelRandomnessR (Output modelA inputA),
    '(modelBRandomness, outputB) ~ ModelRandomnessR (Output modelB inputB),
    HasForwardProduct modelARandomness modelA inputA outputA modelBRandomness modelB inputB outputB
  ) =>
  HasForward (modelA, modelB) (inputA, inputB)
  where
  type
    Output (modelA, modelB) (inputA, inputB) =
      OutputProduct
        (Fst (ModelRandomnessR (Output modelA inputA)))
        modelA
        inputA
        (Snd (ModelRandomnessR (Output modelA inputA)))
        (Fst (ModelRandomnessR (Output modelB inputB)))
        modelB
        inputB
        (Snd (ModelRandomnessR (Output modelB inputB)))
  forward (modelA, modelB) (inputA, inputB) =
    forwardProduct
      (Proxy :: Proxy modelARandomness)
      modelA
      inputA
      (Proxy :: Proxy outputA)
      (Proxy :: Proxy modelBRandomness)
      modelB
      inputB
      (Proxy :: Proxy outputB)

instance
  ( '(modelARandomness, outputA) ~ ModelRandomnessR (Output modelA inputA),
    '(modelBRandomness, outputB) ~ ModelRandomnessR (Output modelB inputB),
    HasForwardSum modelARandomness modelA inputA outputA modelBRandomness modelB inputB outputB
  ) =>
  HasForward (Either modelA modelB) (Either inputA inputB)
  where
  type
    Output (Either modelA modelB) (Either inputA inputB) =
      OutputSum
        (Fst (ModelRandomnessR (Output modelA inputA)))
        modelA
        inputA
        (Snd (ModelRandomnessR (Output modelA inputA)))
        (Fst (ModelRandomnessR (Output modelB inputB)))
        modelB
        inputB
        (Snd (ModelRandomnessR (Output modelB inputB)))
  forward eitherModel eitherIn =
    forwardSum
      (Proxy :: Proxy modelARandomness)
      (Proxy :: Proxy modelBRandomness)
      eitherModel
      eitherIn
      (Proxy :: Proxy (Either outputA outputB))

--
-- Parameterized
--

class Parameterized f where
  flattenParameters :: f -> [Parameter]
  default flattenParameters :: (Generic f, GParameterized (Rep f)) => f -> [Parameter]
  flattenParameters f = gFlattenParameters (from f)

  _replaceParameters :: f -> ParamStream f
  default _replaceParameters :: (Generic f, GParameterized (Rep f)) => f -> ParamStream f
  _replaceParameters f = to <$> _gReplaceParameters (from f)

replaceParameters :: Parameterized f => f -> [Parameter] -> f
replaceParameters f params =
  let (f', remaining) = runState (_replaceParameters f) params
   in if null remaining
        then f'
        else error "Some parameters in a call to replaceParameters haven't been consumed!"

instance Parameterized Tensor where
  flattenParameters _ = []
  _replaceParameters = return

instance Parameterized Parameter where
  flattenParameters = pure
  _replaceParameters _ = nextParameter

instance Parameterized Int where
  flattenParameters _ = []
  _replaceParameters = return

instance Parameterized Float where
  flattenParameters _ = []
  _replaceParameters = return

instance Parameterized Double where
  flattenParameters _ = []
  _replaceParameters = return

instance Parameterized (a -> a) where
  flattenParameters _ = []
  _replaceParameters = return

class GParameterized f where
  gFlattenParameters :: forall a. f a -> [Parameter]
  _gReplaceParameters :: forall a. f a -> ParamStream (f a)

instance GParameterized U1 where
  gFlattenParameters U1 = []
  _gReplaceParameters U1 = return U1

instance (GParameterized f, GParameterized g) => GParameterized (f :+: g) where
  gFlattenParameters (L1 x) = gFlattenParameters x
  gFlattenParameters (R1 x) = gFlattenParameters x
  _gReplaceParameters (L1 x) = do
    x' <- _gReplaceParameters x
    return $ L1 x'
  _gReplaceParameters (R1 x) = do
    x' <- _gReplaceParameters x
    return $ R1 x'

instance (GParameterized f, GParameterized g) => GParameterized (f :*: g) where
  gFlattenParameters (x :*: y) = gFlattenParameters x ++ gFlattenParameters y
  _gReplaceParameters (x :*: y) = do
    x' <- _gReplaceParameters x
    y' <- _gReplaceParameters y
    return $ x' :*: y'

instance (Parameterized c) => GParameterized (K1 i c) where
  gFlattenParameters (K1 x) = flattenParameters x
  _gReplaceParameters (K1 x) = do
    x' <- _replaceParameters x
    return $ K1 x'

instance (GParameterized f) => GParameterized (M1 i t f) where
  gFlattenParameters (M1 x) = gFlattenParameters x
  _gReplaceParameters (M1 x) = do
    x' <- _gReplaceParameters x
    return $ M1 x'

class Randomizable spec f | spec -> f where
  sample :: spec -> IO f

--
-- Linear FC Layer
--

data LinearSpec = LinearSpec
  { in_features :: Int,
    out_features :: Int
  }
  deriving (Show, Eq)

data Linear = Linear
  { weight :: Parameter,
    bias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

instance Parameterized [Linear]

linear :: Linear -> Tensor -> Tensor
linear layer input = linear' input w b
  where
    linear' input weight bias = unsafePerformIO $ (cast3 ATen.linear_ttt) input weight bias
    w = toDependent (weight layer)
    b = toDependent (bias layer)

linearForward :: Linear -> Tensor -> Tensor
linearForward = linear -- temporary alias until dependencies are updated

-- instance HasForward Linear Tensor Tensor where
--   forward = linearForward
--   forwardStoch m x = pure $ linearForward m x

instance Randomizable LinearSpec Linear where
  sample LinearSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [out_features, in_features]
    init <- randIO' [out_features]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ out_features,
                          in_features
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )

    return $ Linear w b

--
-- Conv2d
--

data Conv2dSpec = Conv2dSpec
  { inputChannelSize :: Int,
    outputChannelSize :: Int,
    kernelHeight :: Int,
    kernelWidth :: Int
  }
  deriving (Show, Eq)

data Conv2d = Conv2d
  { conv2dWeight :: Parameter,
    conv2dBias :: Parameter
  }
  deriving (Show, Generic, Parameterized)

conv2dForward :: Conv2d -> (Int, Int) -> (Int, Int) -> Tensor -> Tensor
conv2dForward layer stride padding input =
  Torch.Functional.conv2d' w b stride padding input
  where
    w = toDependent (conv2dWeight layer)
    b = toDependent (conv2dBias layer)

instance Randomizable Conv2dSpec Conv2d where
  sample Conv2dSpec {..} = do
    w <-
      makeIndependent
        =<< kaimingUniform
          FanIn
          (LeakyRelu $ Prelude.sqrt (5.0 :: Float))
          [ outputChannelSize,
            inputChannelSize,
            kernelHeight,
            kernelWidth
          ]
    init <- randIO' [outputChannelSize]
    let bound =
          (1 :: Float)
            / Prelude.sqrt
              ( fromIntegral
                  ( getter FanIn $
                      calculateFan
                        [ outputChannelSize,
                          inputChannelSize,
                          kernelHeight,
                          kernelWidth
                        ]
                  ) ::
                  Float
              )
    b <-
      makeIndependent
        =<< pure
          ( subScalar bound $ mulScalar (bound * 2.0) init
          )

    return $ Conv2d w b
