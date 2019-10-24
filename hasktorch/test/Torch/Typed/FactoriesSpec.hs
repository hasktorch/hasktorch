{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.FactoriesSpec
  ( Torch.Typed.FactoriesSpec.spec
  )
where

import           Control.Exception.Safe
import           Foreign.Storable
import           Data.HList
import           Data.Proxy
import           Data.Reflection

import           Test.Hspec
import           Test.QuickCheck

import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Factories
import           Torch.Typed.Native
import           Torch.Typed.Tensor
import           Torch.Typed.AuxSpec

data SimpleFactoriesSpec = ZerosSpec | OnesSpec | RandSpec | RandnSpec

instance (TensorOptions shape dtype device)
  => Apply
       SimpleFactoriesSpec
       (Proxy '(device, dtype, shape))
       (() -> IO ())
 where
  apply ZerosSpec _ _ = do
    let t = zeros :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply OnesSpec _ _ = do
    let t = ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply RandSpec _ _ = do
    t <- rand :: IO (Tensor device dtype shape)
    checkDynamicTensorAttributes t
  apply RandnSpec _ _ = do
    t <- randn :: IO (Tensor device dtype shape)
    checkDynamicTensorAttributes t

spec :: Spec
spec = do
  describe "simple factories" $ do
    it "ones" (hfoldrM @IO ZerosSpec () (allDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "zeros" (hfoldrM @IO OnesSpec () (allDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "rand" (hfoldrM @IO RandSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
    it "randn" (hfoldrM @IO RandnSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
  describe "advanced factories" $ do
    it "linspace" $ do
      let t = linspace @3 @'( 'D.CPU, 0) (1 :: Int) (3 :: Int)
      checkDynamicTensorAttributes t
    it "eyeSquare" $ do
      let t = eyeSquare @10 @'D.Float @'( 'D.CPU, 0)
      checkDynamicTensorAttributes t
