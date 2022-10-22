{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Model where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import GHC.Generics (Generic)
import Torch.GraduallyTyped

newtype NeuralInterpreter transformer = NeuralInterpreter { unNeuralInterpreter :: transformer }
  deriving stock (Eq, Ord, Show, Generic)

type instance ModelSpec (NeuralInterpreter transformer) = NeuralInterpreter (ModelSpec transformer)

instance
  HasInitialize transformer generatorDevice transformer' generatorOutputDevice =>
  HasInitialize (NeuralInterpreter transformer) generatorDevice (NeuralInterpreter transformer') generatorOutputDevice

instance HasStateDict transformer => HasStateDict (NeuralInterpreter transformer)

instance
  ( HasForward
      transformer
      (SimplifiedEncoderDecoderTransformerTrainingInput input target)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerTrainingOutput loss)
      generatorOutputDevice,
    loss ~ Tensor lossGradient lossLayout lossDevice lossDataType lossShape
  ) =>
  HasForward
    (NeuralInterpreter transformer)
    (input, target)
    generatorDevice
    loss
    generatorOutputDevice
  where
  forward (NeuralInterpreter transformer) (sedtTrainingInput, sedtTarget) =
    runIxStateT $
      ireturn SimplifiedEncoderDecoderTransformerTrainingInput {..}
        >>>= IxStateT . forward transformer
        >>>= ireturn . sedtLoss
