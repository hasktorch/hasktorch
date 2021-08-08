{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Examples.NeuralInterpreter.Model where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import GHC.Generics (Generic)
import Torch.GraduallyTyped

data NeuralInterpreter transformer logSoftmax
  = NeuralInterpreter transformer logSoftmax
  deriving stock (Generic)

type instance
  ModelSpec (NeuralInterpreter transformer logSoftmax) =
    NeuralInterpreter (ModelSpec transformer) (ModelSpec logSoftmax)

instance
  ( HasInitialize transformer generatorDevice transformer' generatorDevice',
    HasInitialize logSoftmax generatorDevice' logSoftmax' generatorOutputDevice
  ) =>
  HasInitialize
    (NeuralInterpreter transformer logSoftmax)
    generatorDevice
    (NeuralInterpreter transformer' logSoftmax')
    generatorOutputDevice

instance
  (HasStateDict transformer, HasStateDict logSoftmax) =>
  HasStateDict (NeuralInterpreter transformer logSoftmax)

instance
  ( HasForward
      transformer
      (SimplifiedEncoderDecoderTransformerInput input target)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput inputPaddingMask)
      generatorDevice',
    HasForward
      logSoftmax
      decoderOutput
      generatorDevice'
      ()
      generatorOutputDevice
  ) =>
  HasForward
    (NeuralInterpreter transformer logSoftmax)
    (SimplifiedEncoderDecoderTransformerInput input target)
    generatorDevice
    ()
    generatorOutputDevice
  where
  forward
    (NeuralInterpreter transformer logSoftmax)
    input@SimplifiedEncoderDecoderTransformerInput {..} =
      runIxStateT $
        ireturn input
          >>>= IxStateT . forward transformer
          >>>= ireturn . (\SimplifiedEncoderDecoderTransformerOutput {..} -> sedtDecoderOutput)
          >>>= IxStateT . forward logSoftmax