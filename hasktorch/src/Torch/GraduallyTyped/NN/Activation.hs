{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Activation where

import Data.Singletons.TypeLits (Nat, Symbol)
import GHC.Generics (Generic)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.Shape (Size, Name, Dim, WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

data Softmax (dim :: Dim (Name Symbol) (Size Nat)) where
  Softmax ::
    forall dim.
    Dim String Integer ->
    Softmax dim
  deriving (Generic)

instance WithDimC dim (Softmax dim) => HasInitialize (Softmax dim) where
  type InitializeF (Softmax dim) = WithDimF dim (Softmax dim)
  initialize = withDim @dim $ Softmax @dim

instance
  HasForward
    (Softmax dim)
    (Tensor requiresGradient layout device dataType shape)
  where
  type
    ForwardOutput
      (Softmax dim)
      (Tensor requiresGradient layout device dataType shape) =
      Tensor requiresGradient layout device dataType shape
  forward _ = undefined
