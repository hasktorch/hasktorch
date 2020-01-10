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
import           Torch.HList
import           Data.Proxy
import           Data.Reflection
import           GHC.TypeLits

import           Test.Hspec
import           Test.QuickCheck

import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functional               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor
import           Torch.Typed.AuxSpec

data SimpleFactoriesSpec = ZerosSpec | OnesSpec

instance (TensorOptions shape dtype device)
  => Apply
       SimpleFactoriesSpec
       (Proxy device, (Proxy dtype, Proxy shape))
       (() -> IO ())
 where
  apply ZerosSpec _ _ = do
    let t = zeros :: Tensor device dtype shape
    checkDynamicTensorAttributes t
  apply OnesSpec _ _ = do
    let t = ones :: Tensor device dtype shape
    checkDynamicTensorAttributes t


data RandomFactoriesSpec = RandSpec | RandnSpec

instance ( TensorOptions shape dtype device
         , RandDTypeIsValid device dtype
         )
  => Apply
       RandomFactoriesSpec
       (Proxy device, (Proxy dtype, Proxy shape))
       (() -> IO ())
 where
  apply RandSpec _ _ = do
    t <- rand :: IO (Tensor device dtype shape)
    checkDynamicTensorAttributes t
  apply RandnSpec _ _ = do
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
              D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
                hfoldrM @IO simpleFactoriesSpec () (hattach cpu   (hproduct allDTypes standardShapes))
              D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
                hfoldrM @IO simpleFactoriesSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "ones"  $ dispatch ZerosSpec
      it "zeros" $ dispatch OnesSpec
    describe "random factories" $ do
      let dispatch randomFactoriesSpec = 
            case device of
              D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } ->
                hfoldrM @IO randomFactoriesSpec () (hattach cpu   (hproduct standardFloatingPointDTypes standardShapes))
              D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } ->
                hfoldrM @IO randomFactoriesSpec () (hattach cuda0 (hproduct allFloatingPointDTypes      standardShapes))
      it "rand"  $ dispatch RandSpec
      it "randn" $ dispatch RandnSpec
    describe "advanced factories" $ do
      it "linspace" $ case device of
        D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } -> do
          let t = linspace @3 @'( 'D.CPU, 0)  (1 :: Int) (3 :: Int)
          checkDynamicTensorAttributes t
        D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } -> do
          let t = linspace @3 @'( 'D.CUDA, 0) (1 :: Int) (3 :: Int)
          checkDynamicTensorAttributes t
      it "eyeSquare" $ case device of
        D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 } -> do
          let t = eyeSquare @10 @'D.Float @'( 'D.CPU, 0)
          checkDynamicTensorAttributes t
        D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 } -> do
          let t = eyeSquare @10 @'D.Float @'( 'D.CUDA, 0)
          checkDynamicTensorAttributes t
