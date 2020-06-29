{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NN.Normalization where

import           Control.Monad.State.Strict
import           Torch.HList
import           Data.Kind                    (Type)
import           Data.Proxy
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.Generics
import           System.IO.Unsafe

import qualified Torch.NN                      as A
import           Torch.NN                     (HasForward(..))
import qualified Torch.Autograd                as A
import qualified Torch.Tensor                  as A
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Device

data LayerNormSpec (normalizedShape :: [Nat]) (dtype :: D.DType) (device :: (D.DeviceType, Nat))
 where
  LayerNormSpec
    :: forall normalizedShape dtype device
     . { layerNormEpsSpec :: Double}
    -> LayerNormSpec normalizedShape dtype device
 deriving (Show, Eq)

data LayerNorm (normalizedShape :: [Nat]) (dtype :: D.DType) (device :: (D.DeviceType, Nat))
 where
  LayerNorm
    :: { layerNormWeight :: Parameter device dtype normalizedShape
       , layerNormBias   :: Parameter device dtype normalizedShape
       , layerNormEps    :: Double
       }
    -> LayerNorm normalizedShape dtype device
 deriving (Show, Generic)

layerNormForward
  :: forall normalizedShape shape dtype device
   . ( EndsWith shape normalizedShape
     , KnownShape normalizedShape
     )
  => LayerNorm normalizedShape dtype device
  -> Tensor device dtype shape
  -> Tensor device dtype shape
layerNormForward LayerNorm {..} = layerNorm @normalizedShape
  (toDependent layerNormWeight)
  (toDependent layerNormBias)
  layerNormEps

instance
  ( EndsWith shape normalizedShape
  , KnownShape normalizedShape
  ) => HasForward (LayerNorm normalizedShape dtype device) (Tensor device dtype shape) (Tensor device dtype shape) where
  forward = layerNormForward

instance
  ( TensorOptions normalizedShape dtype device
  , RandDTypeIsValid device dtype
  ) => A.Randomizable (LayerNormSpec normalizedShape dtype device)
                      (LayerNorm     normalizedShape dtype device)
 where
  sample LayerNormSpec {..} =
    LayerNorm
      <$> (makeIndependent =<< randn)
      <*> (makeIndependent =<< randn)
      <*> pure layerNormEpsSpec
