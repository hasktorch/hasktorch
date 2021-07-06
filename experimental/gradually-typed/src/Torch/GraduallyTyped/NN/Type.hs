{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}

module Torch.GraduallyTyped.NN.Type where

import GHC.Generics (Generic)

data HasBias = WithBias | WithoutBias
  deriving stock (Eq, Ord, Show, Generic)
