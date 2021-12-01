{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Typed.FactoriesSpec
  ( Torch.Typed.FactoriesSpec.spec,
  )
where

import Data.Proxy
import Test.Hspec (Spec, describe, it)
import Test.QuickCheck ()
import Torch.Typed
import Torch.Typed.AuxiliarySpec

data SimpleFactoriesSpec = ZerosSpec | OnesSpec | FullSpec

instance
  ( TensorOptions shape dtype device
  ) =>
  Apply' SimpleFactoriesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' ZerosSpec (_, agg) =
    agg >> do
      let t = zeros :: Tensor device dtype shape
      checkDynamicTensorAttributes t
  apply' OnesSpec (_, agg) =
    agg >> do
      let t = ones :: Tensor device dtype shape
      checkDynamicTensorAttributes t
  apply' FullSpec (_, agg) =
    agg >> do
      let t = full (2.0 :: Float) :: Tensor device dtype shape
      checkDynamicTensorAttributes t

data RandomFactoriesSpec = RandSpec | RandnSpec

instance
  ( TensorOptions shape dtype device,
    RandDTypeIsValid device dtype
  ) =>
  Apply' RandomFactoriesSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' RandSpec (_, agg) =
    agg >> do
      t <- rand :: IO (Tensor device dtype shape)
      checkDynamicTensorAttributes t
  apply' RandnSpec (_, agg) =
    agg >> do
      t <- randn :: IO (Tensor device dtype shape)
      checkDynamicTensorAttributes t

spec :: Spec
spec = foldMap spec' availableDevices

spec' :: Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
    describe "simple factories" $ do
      let dispatch simpleFactoriesSpec =
            case device of
              Device {deviceType = CPU, deviceIndex = 0} ->
                hfoldrM @IO simpleFactoriesSpec () (hattach cpu (hproduct allDTypes standardShapes))
              Device {deviceType = CUDA, deviceIndex = 0} ->
                hfoldrM @IO simpleFactoriesSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "ones" $ dispatch ZerosSpec
      it "zeros" $ dispatch OnesSpec
      it "full" $ dispatch FullSpec
    describe "random factories" $ do
      let dispatch randomFactoriesSpec =
            case device of
              Device {deviceType = CPU, deviceIndex = 0} ->
                hfoldrM @IO randomFactoriesSpec () (hattach cpu (hproduct standardFloatingPointDTypes standardShapes))
              Device {deviceType = CUDA, deviceIndex = 0} ->
                hfoldrM @IO randomFactoriesSpec () (hattach cuda0 (hproduct allFloatingPointDTypes standardShapes))
      it "rand" $ dispatch RandSpec
      it "randn" $ dispatch RandnSpec
    describe "advanced factories" $ do
      it "linspace" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} -> do
          let t = linspace @3 @'( 'CPU, 0) (1 :: Int) (3 :: Int)
          checkDynamicTensorAttributes t
        Device {deviceType = CUDA, deviceIndex = 0} -> do
          let t = linspace @3 @'( 'CUDA, 0) (1 :: Int) (3 :: Int)
          checkDynamicTensorAttributes t
      it "eyeSquare" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} -> do
          let t = eyeSquare @10 @'Float @'( 'CPU, 0)
          checkDynamicTensorAttributes t
        Device {deviceType = CUDA, deviceIndex = 0} -> do
          let t = eyeSquare @10 @'Float @'( 'CUDA, 0)
          checkDynamicTensorAttributes t
