{-# LANGUAGE TypeApplications #-}
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
import Torch.GraduallyTyped.Prelude (Contains, ErrorMessage (Text), Fst, If, Proxy (..), Snd, Type, TypeError)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Tensor (Device (AnyDevice))

data ModelRandomness = Deterministic | Stochastic

class HasForward model input where
  type Output model input :: Type
  forward :: model -> input -> Output model input

type family ModelRandomnessR (output :: Type) :: (ModelRandomness, Type) where
  ModelRandomnessR (Generator device -> (output, Generator device)) =
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
    outputA
    (modelBRandomness :: ModelRandomness)
    outputB
    modelA
    inputA
    modelB
    inputB
  where
  type
    OutputProduct modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB ::
      Type
  forwardProduct ::
    Proxy modelARandomness ->
    Proxy outputA ->
    Proxy modelBRandomness ->
    Proxy outputB ->
    (modelA, modelB) ->
    (inputA, inputB) ->
    OutputProduct modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ (Generator 'AnyDevice -> (outputA, Generator 'AnyDevice)),
    HasForward modelB inputB,
    Output modelB inputB ~ (Generator 'AnyDevice -> (outputB, Generator 'AnyDevice))
  ) =>
  HasForwardProduct 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB
  where
  type
    OutputProduct 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB =
      Generator 'AnyDevice -> ((outputA, outputB), Generator 'AnyDevice)
  forwardProduct _ _ _ _ (modelA, modelB) (inputA, inputB) =
    runState $ do
      outputA <- state (forward modelA inputA)
      outputB <- state (forward modelB inputB)
      return (outputA, outputB)

instance
  ( '(modelARandomness, outputA) ~ ModelRandomnessR (Output modelA inputA),
    '(modelBRandomness, outputB) ~ ModelRandomnessR (Output modelB inputB),
    HasForwardProduct modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB
  ) =>
  HasForward (modelA, modelB) (inputA, inputB)
  where
  type
    Output (modelA, modelB) (inputA, inputB) =
      OutputProduct
        (Fst (ModelRandomnessR (Output modelA inputA)))
        (Snd (ModelRandomnessR (Output modelA inputA)))
        (Fst (ModelRandomnessR (Output modelB inputB)))
        (Snd (ModelRandomnessR (Output modelB inputB)))
        modelA
        inputA
        modelB
        inputB
  forward =
    forwardProduct
      (Proxy :: Proxy modelARandomness)
      (Proxy :: Proxy outputA)
      (Proxy :: Proxy modelBRandomness)
      (Proxy :: Proxy outputB)

class
  HasForwardSum
    (modelARandomness :: ModelRandomness)
    outputA
    (modelBRandomness :: ModelRandomness)
    outputB
    modelA
    inputA
    modelB
    inputB
  where
  type
    OutputSum modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB ::
      Type
  forwardSum ::
    Proxy modelARandomness ->
    Proxy outputA ->
    Proxy modelBRandomness ->
    Proxy outputB ->
    Either modelA modelB ->
    Either inputA inputB ->
    OutputSum modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB

instance
  ( HasForward modelA inputA,
    Output modelA inputA ~ (Generator 'AnyDevice -> (outputA, Generator 'AnyDevice)),
    HasForward modelB inputB,
    Output modelB inputB ~ (Generator 'AnyDevice -> (outputB, Generator 'AnyDevice))
  ) =>
  HasForwardSum 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB
  where
  type
    OutputSum 'Stochastic outputA 'Stochastic outputB modelA inputA modelB inputB =
      Generator 'AnyDevice -> (Maybe (Either outputA outputB), Generator 'AnyDevice)
  forwardSum _ _ _ _ (Left modelA) (Left inputA) =
    runState $ Just . Left <$> (state $ forward modelA inputA)
  forwardSum _ _ _ _ (Right modelB) (Right inputB) =
    runState $ Just . Right <$> (state $ forward modelB inputB)
  forwardSum _ _ _ _ _ _ = runState . pure $ Nothing

instance
  ( '(modelARandomness, outputA) ~ ModelRandomnessR (Output modelA inputA),
    '(modelBRandomness, outputB) ~ ModelRandomnessR (Output modelB inputB),
    HasForwardSum modelARandomness outputA modelBRandomness outputB modelA inputA modelB inputB
  ) =>
  HasForward (Either modelA modelB) (Either inputA inputB)
  where
  type
    Output (Either modelA modelB) (Either inputA inputB) =
      OutputSum
        (Fst (ModelRandomnessR (Output modelA inputA)))
        (Snd (ModelRandomnessR (Output modelA inputA)))
        (Fst (ModelRandomnessR (Output modelB inputB)))
        (Snd (ModelRandomnessR (Output modelB inputB)))
        modelA
        inputA
        modelB
        inputB
  forward =
    forwardSum
      (Proxy :: Proxy modelARandomness)
      (Proxy :: Proxy outputA)
      (Proxy :: Proxy modelBRandomness)
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

test' :: Generator 'AnyDevice -> (Maybe (Either OutputA OutputB), Generator 'AnyDevice)
test' = forward @(Either ModelA ModelB) @(Either InputA InputB) (Left ModelA) (Right InputB)
