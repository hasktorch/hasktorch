{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NN.Linear where

import           GHC.TypeLits
import           GHC.Generics

import           Torch.NN                     (Randomizable(..), HasForward(..))
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter

data
  LinearSpec
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = LinearSpec deriving (Show, Eq)

data
  Linear
    (inputFeatures :: Nat)
    (outputFeatures :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat)) where
  Linear
    :: forall inputFeatures outputFeatures dtype device
     . { linearWeight :: Parameter device dtype '[outputFeatures, inputFeatures]
       , linearBias   :: Parameter device dtype '[outputFeatures]
       }
    -> Linear inputFeatures outputFeatures dtype device
  deriving (Show, Generic)

-- | linear
-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
linearForward
  :: _
  => Linear _ _ _ _
  -> Tensor _ _ _
  -> Tensor _ _ _
linearForward Linear {..} input = linear' (toDependent linearWeight) (toDependent linearBias) input

instance
  ( shape'' ~ MatMul shape '[inputFeatures, outputFeatures]
  , shape' ~ Broadcast shape'' shape''
  ) => HasForward (Linear inputFeatures outputFeatures dtype device) (Tensor device dtype shape) (Tensor device dtype shape') where
  forward = linearForward

instance
  ( KnownNat inputFeatures
  , KnownNat outputFeatures
  , KnownDType dtype
  , KnownDevice device
  , RandDTypeIsValid device dtype
  ) => Randomizable (LinearSpec inputFeatures outputFeatures dtype device)
                    (Linear     inputFeatures outputFeatures dtype device)
 where
  sample LinearSpec =
    Linear <$> (makeIndependent =<< randn) <*> (makeIndependent =<< randn)
