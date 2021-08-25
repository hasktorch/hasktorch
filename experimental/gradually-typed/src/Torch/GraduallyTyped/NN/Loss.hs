{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Loss where

import GHC.Generics (Generic)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Functional.Loss (mseLoss)
import Torch.GraduallyTyped.Prelude (Catch)
import Torch.GraduallyTyped.Shape.Type (Shape (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

data MSELoss = MSELoss
  deriving stock (Eq, Ord, Show, Generic)

type instance ModelSpec MSELoss = MSELoss

instance HasInitialize MSELoss generatorDevice MSELoss generatorDevice

instance HasStateDict MSELoss

instance
  ( Catch (predShape <+> targetShape),
    output
      ~ Tensor
          (predGradient <|> targetGradient)
          (predLayout <+> targetLayout)
          (predDevice <+> targetDevice)
          (predDataType <+> targetDataType)
          ('Shape '[])
  ) =>
  HasForward
    MSELoss
    ( Tensor predGradient predLayout predDevice predDataType predShape,
      Tensor targetGradient targetLayout targetDevice targetDataType targetShape
    )
    generatorDevice
    output
    generatorDevice
  where
  forward MSELoss (prediction, target) g = do
    loss <- prediction `mseLoss` target
    pure (loss, g)
