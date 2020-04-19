{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.FactoriesSpec
  ( Torch.Typed.FactoriesSpec.spec,
  )
where

import Control.Exception.Safe
import Data.Proxy
import Data.Reflection
import Foreign.Storable
import GHC.TypeLits
import Test.Hspec
import Test.QuickCheck
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as D
import Torch.HList
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.TensorOptions as D
import Torch.Typed.AuxSpec
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Tensor

data SimpleFactoriesSpec = ZerosSpec | OnesSpec | FullSpec

instance
  ( TensorOptions shape dtype device
  ) =>
  Apply' SimpleFactoriesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' ZerosSpec (_, agg) = agg >> do
    let t = zeros :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply' OnesSpec (_, agg) = agg >> do
    let t = ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply' FullSpec (_, agg) = agg >> do
    let t = full (2.0 :: Float) :: Tensor device dtype shape
    checkDynamicTensorAttributes t

data RandomFactoriesSpec = RandSpec | RandnSpec

instance
  ( TensorOptions shape dtype device,
    RandDTypeIsValid device dtype
  ) =>
  Apply' RandomFactoriesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' RandSpec (_, agg) = agg >> do
    t <- rand :: IO (Tensor device dtype shape)
    checkDynamicTensorAttributes t
  apply' RandnSpec (_, agg) = agg >> do
    t <- randn :: IO (Tensor device dtype shape)
    checkDynamicTensorAttributes t

spec = foldMap spec' availableDevices

spec' :: D.Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
    describe "simple factories" $ do
      let dispatch simpleFactoriesSpec =
            case device of
              D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
                hfoldrM @IO simpleFactoriesSpec () (hattach cpu (hproduct allDTypes standardShapes))
              D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
                hfoldrM @IO simpleFactoriesSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "ones" $ dispatch ZerosSpec
      it "zeros" $ dispatch OnesSpec
      it "full" $ dispatch FullSpec
    describe "random factories" $ do
      let dispatch randomFactoriesSpec =
            case device of
              D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} ->
                hfoldrM @IO randomFactoriesSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
              D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} ->
                hfoldrM @IO randomFactoriesSpec () (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes))
      it "rand" $ dispatch RandSpec
      it "randn" $ dispatch RandnSpec
    describe "advanced factories" $ do
      it "linspace" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} -> do
          let t = linspace @3 @'( 'D.CPU, 0) (1 :: Int) (3 :: Int)
          checkDynamicTensorAttributes t
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} -> do
          let t = linspace @3 @'( 'D.CUDA, 0) (1 :: Int) (3 :: Int)
          checkDynamicTensorAttributes t
      it "eyeSquare" $ case device of
        D.Device {D.deviceType = D.CPU, D.deviceIndex = 0} -> do
          let t = eyeSquare @10 @'D.Float @'( 'D.CPU, 0)
          checkDynamicTensorAttributes t
        D.Device {D.deviceType = D.CUDA, D.deviceIndex = 0} -> do
          let t = eyeSquare @10 @'D.Float @'( 'D.CUDA, 0)
          checkDynamicTensorAttributes t
