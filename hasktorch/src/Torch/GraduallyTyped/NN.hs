{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Type.Bool (If, type (||))
import GHC.TypeLits (ErrorMessage (Text), TypeError (..))
import Torch.GraduallyTyped.Prelude (Fst, Snd)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Tensor (Device (AnyDevice))

data ModelRandomness = Deterministic | Stochastic

class HasForward model input where
  type Output model input :: Type
  forward :: model -> input -> Output model input

type family Contains (f :: k) (a :: k') :: Bool where
  Contains a a = 'True
  Contains (f g) a = Contains f a || Contains g a
  Contains _ _ = 'False

type family ModelRandomnessR (output :: Type) :: (ModelRandomness, Type) where
  ModelRandomnessR ((Generator device) -> (output, (Generator device))) =
    If
      (Contains output Generator)
      (TypeError (Text "The random generator appears in a wrong position in the output type."))
      '( 'Stochastic, output)
  ModelRandomnessR output =
    If
      (Contains output Generator)
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

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ (Generator 'AnyDevice -> (outputA, Generator 'AnyDevice)),
    HasForward modelB inputB,
    Output modelB inputB ~ (Generator 'AnyDevice -> (outputB, Generator 'AnyDevice))
  ) =>
  HasForwardProduct 'Stochastic modelA inputA outputA 'Stochastic modelB inputB outputB
  where
  type
    OutputProduct 'Stochastic modelA inputA outputA 'Stochastic modelB inputB outputB =
      Generator 'AnyDevice ->
      ((outputA, outputB), Generator 'AnyDevice)
  forwardProduct _ modelA inputA _ _ modelB inputB _ = runState $ do
    outputA <- state (forward modelA inputA)
    outputB <- state (forward modelB inputB)
    return (outputA, outputB)

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

data ModelA = ModelA

data InputA = InputA

data OutputA = OutputA

instance HasForward ModelA InputA where
  type Output ModelA InputA = (Generator 'AnyDevice -> (OutputA, Generator 'AnyDevice))
  forward _ _ g = (OutputA, g)

data ModelB = ModelB

data InputB = InputB

data OutputB = OutputB

instance HasForward ModelB InputB where
  type Output ModelB InputB = (Generator 'AnyDevice -> (OutputB, Generator 'AnyDevice))
  forward _ _ g = (OutputB, g)

test :: Generator 'AnyDevice -> ((OutputA, OutputB), Generator 'AnyDevice)
test = forward (ModelA, ModelB) (InputA, InputB)
