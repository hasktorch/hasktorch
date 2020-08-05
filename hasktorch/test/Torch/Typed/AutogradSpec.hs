{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.AutogradSpec
  ( Torch.Typed.AutogradSpec.spec
  )
where

import Control.Monad    (when)
import Data.Kind        ()
import Data.Maybe       ()
import Data.Proxy
import Data.Reflection  ()
import GHC.Exts         (toList)
import GHC.Generics
import GHC.TypeLits
import Prelude          hiding (all, cos, sin)
import System.IO.Unsafe ()

import Test.Hspec      (Spec, describe, it, shouldBe)
import Test.QuickCheck ()

import Torch                (ATenTensor)
import Torch.Internal.Class (Castable)
import Torch.Typed
import Torch.Typed.AuxSpec

data RastriginLayerSpec (n :: Nat) (dtype :: DType) (device ::
                                                  (DeviceType, Nat)) = RastriginLayerSpec
    deriving (Show, Eq)

data RastriginLayer (n :: Nat)
                    (dtype :: DType)
                    (device :: (DeviceType, Nat))
  where
    RastriginLayer
      :: forall n dtype device
       . { x :: Parameter device dtype '[n] }
      -> RastriginLayer n dtype device
  deriving (Show, Generic)

instance
  ( RandDTypeIsValid device dtype
  , KnownNat n
  , KnownDType dtype
  , KnownDevice device
  ) => Randomizable (RastriginLayerSpec n dtype device)
                    (RastriginLayer     n dtype device)
 where
  sample _ = RastriginLayer <$> (makeIndependent =<< randn)

rastriginLayer'
  :: forall device dtype a n shape
   . ( SumDTypeIsValid device dtype
     , StandardFloatingPointDTypeValidation device dtype
     , All Scalar '[a, n]
     , KnownDType (SumDType dtype)
     , KnownDevice device
     )
  => Tensor device dtype shape
  -> a
  -> n
  -> Tensor device (SumDType dtype) '[]
rastriginLayer' x a n = (mulScalar a . mulScalar n $ ones)
  + sumAll (x * x - (mulScalar a . cos . mulScalar (2 * pi :: Double)) x)

gradientsRastriginLayer'
  :: forall device dtype a shape
   . (KnownDevice device, StandardFloatingPointDTypeValidation device dtype, Scalar a)
  => Tensor device dtype shape
  -> a
  -> Tensor device dtype shape
gradientsRastriginLayer' x a = mulScalar
  (2 :: Int)
  (x + (mulScalar a . mulScalar (pi :: Double) . sin . mulScalar (2 * pi :: Double)) x)

data RastriginStackSpec (num :: Nat) (ns :: [Nat]) (dtypes ::
                                                 [DType]) (devices :: [(DeviceType, Nat)]) = RastriginStackSpec
    deriving (Show, Eq)

data RastriginStack (num :: Nat)
                    (ns :: [Nat])
                    (dtypes :: [DType])
                    (devices :: [(DeviceType, Nat)])
 where
  Rastrigin1
    :: forall n dtype device
     . RastriginLayer n dtype device
    -> RastriginStack 1 '[n] '[dtype] '[device]
  RastriginK
    :: forall num n ns dtype dtypes device devices
     . (2 <= num)
    => RastriginStack (num - 1) ns dtypes devices
    -> RastriginLayer n dtype device
    -> RastriginStack num (n ': ns) (dtype ': dtypes) (device ': devices)

deriving instance Show (RastriginStack num ns dtypes devices)

instance {-# OVERLAPS #-}
  ( layer ~ (K1 R (RastriginLayer n dtype device))
  , GParameterized layer parameters
  ) => GParameterized (K1 R (RastriginStack 1 '[n] '[dtype] '[device])) parameters where
  gFlattenParameters (K1 (Rastrigin1 rastrigin))
    = gFlattenParameters (K1 rastrigin :: layer _)
  gReplaceParameters (K1 (Rastrigin1 rastrigin)) parameters
    = K1 (Rastrigin1 (unK1 (gReplaceParameters (K1 rastrigin :: layer _) parameters)))

instance {-# OVERLAPPABLE #-}
  ( 2 <= num
  , layerStack ~ (K1 R (RastriginStack (num - 1) ns dtypes devices))
  , layer ~ (K1 R (RastriginLayer n dtype device))
  , GParameterized layerStack parameters
  , GParameterized layer parameters'
  , HAppendFD parameters parameters' parameters''
  , parameters'' ~ (parameters ++ parameters')
  ) => GParameterized (K1 R (RastriginStack num (n ': ns) (dtype ': dtypes) (device ': devices))) parameters'' where
  gFlattenParameters (K1 (RastriginK rastriginStack rastriginLayer))
    = let parameters  = gFlattenParameters (K1 rastriginStack :: layerStack _)
          parameters' = gFlattenParameters (K1 rastriginLayer :: layer _)
      in  parameters `happendFD` parameters'
  gReplaceParameters (K1 (RastriginK rastriginStack rastriginLayer)) parameters''
    = let (parameters, parameters') = hunappendFD parameters''
          rastriginStack'           = unK1 (gReplaceParameters (K1 rastriginStack :: layerStack _) parameters)
          rastriginLayer'           = unK1 (gReplaceParameters (K1 rastriginLayer :: layer _) parameters')
      in  K1 (RastriginK rastriginStack' rastriginLayer')

instance {-# OVERLAPS #-}
  ( RandDTypeIsValid device dtype
  , KnownNat n
  , KnownDType dtype
  , KnownDevice device
  ) => Randomizable (RastriginStackSpec 1 '[n] '[dtype] '[device])
                    (RastriginStack     1 '[n] '[dtype] '[device])
 where
  sample _ = Rastrigin1 <$> (sample $ RastriginLayerSpec @n @dtype @device)

instance {-# OVERLAPPABLE #-}
  ( 2 <= num
  , RandDTypeIsValid device dtype
  , KnownNat n
  , KnownDType dtype
  , KnownDevice device
  , Randomizable (RastriginStackSpec (num - 1) ns dtypes devices)
                 (RastriginStack     (num - 1) ns dtypes devices)
  ) => Randomizable (RastriginStackSpec num (n ': ns) (dtype ': dtypes) (device ': devices))
                    (RastriginStack     num (n ': ns) (dtype ': dtypes) (device ': devices))
 where
  sample _ =
    RastriginK
      <$> (sample $ RastriginStackSpec @(num - 1) @ns @dtypes @devices)
      <*> (sample $ RastriginLayerSpec @n @dtype @device)

data RastriginSpec (num :: Nat) (ns :: [Nat]) (dtypes ::
                                            [DType]) (devices :: [(DeviceType, Nat)]) = RastriginSpec
    deriving (Show, Eq)

data Rastrigin (num :: Nat) (ns :: [Nat]) (dtypes :: [DType]) (devices
                                                            :: [(DeviceType, Nat)]) = Rastrigin
    { rastriginStack :: RastriginStack num ns dtypes devices
    }
    deriving (Show, Generic)

instance
  ( Randomizable (RastriginStackSpec num ns dtypes devices)
                 (RastriginStack     num ns dtypes devices)
  ) => Randomizable (RastriginSpec num ns dtypes devices)
                    (Rastrigin     num ns dtypes devices)
 where
  sample _ =
    Rastrigin <$> (sample $ RastriginStackSpec @num @ns @dtypes @devices)

data RastriginA a dv dt = RastriginA a dv dt

instance
  ( Scalar a
  , KnownNat n
  , All KnownDType [SumDType dtype, dtype']
  , All KnownDevice [device, device']
  , SumDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device dtype
  ) => Apply' (RastriginA a (Proxy device') (Proxy dtype')) (Parameter device dtype '[n]) (Tensor device' dtype' '[]) where
  apply' (RastriginA a _ _) parameter =
    toDevice @device' . toDType @dtype' @(SumDType dtype) . rastriginLayer' (toDependent parameter) a $ natValI @n

rastrigin
  :: forall a dtype device tensors parameters num ns dtypes devices shape
   . ( SumDType dtype ~ dtype
     , SumDTypeIsValid device dtype
     , Parameterized (Rastrigin num ns dtypes devices) parameters
     , HMap' (RastriginA a (Proxy device) (Proxy dtype)) parameters tensors
     , Castable (HList tensors) [ATenTensor]
     , '(shape, dtype, device) ~ Stack 0 tensors
     , DropValue shape 0 ~ '[]
     )
  => Rastrigin num ns dtypes devices
  -> a
  -> Tensor device dtype '[]
rastrigin model a =
  sumDim @0
    . stack @0
    . hmap' (RastriginA a (Proxy @device) (Proxy @dtype))
    . flattenParameters
    $ model

data GradientsRastriginA a = GradientsRastriginA a

instance
  ( KnownDevice device
  , StandardFloatingPointDTypeValidation device dtype
  , Scalar a
  ) => Apply' (GradientsRastriginA a) (Parameter device dtype '[n]) (Tensor device dtype '[n]) where
  apply' (GradientsRastriginA a) parameter = gradientsRastriginLayer' (toDependent parameter) $ a

gradientsRastrigin
  :: forall gradients a num ns dtypes devices parameters
   . ( HMap' (GradientsRastriginA a) parameters gradients
     , Parameterized (Rastrigin num ns dtypes devices) parameters
     )
  => Rastrigin num ns dtypes devices
  -> a
  -> HList gradients
gradientsRastrigin model a = hmap' (GradientsRastriginA a) . flattenParameters $ model

data GradientsTestInner = GradientsTestInner

instance
  ( TensorOptions shape dtype device
  ) => Apply' GradientsTestInner ((Tensor device dtype shape, Tensor device dtype shape), IO ()) (IO ()) where
  apply' _ ((a, b), agg) = agg >> do
    checkDynamicTensorAttributes a
    checkDynamicTensorAttributes b
    (toList . Just . toDevice @'( 'CPU, 0) @device . unsqueeze @0 . all)
        (isclose 1e-05 1e-08 False a b)
      `shouldBe` [True]

data GradientsTestOuter a = GradientsTestOuter a

instance
  ( Randomizable (RastriginStackSpec num ns dtypes devices) (RastriginStack num ns dtypes devices)
  , HasGrad (HList parameters) (HList gradients)
  , SumDType dtype ~ dtype
  , SumDTypeIsValid device dtype
  , Parameterized (Rastrigin num ns dtypes devices) parameters
  , HMap' (RastriginA a (Proxy device) (Proxy dtype)) parameters tensors
  , Castable (HList tensors) [ATenTensor]
  , '(shape, dtype, device) ~ Stack 0 tensors
  , DropValue shape 0 ~ '[]
  , HMap' (GradientsRastriginA a) parameters gradients'
  , HZip gradients gradients' zs
  , HFoldrM IO GradientsTestInner () zs ()
  ) => Apply' (GradientsTestOuter a) (((Proxy device, Proxy dtype), RastriginSpec num ns dtypes devices), IO ()) (IO ()) where
  apply' (GradientsTestOuter a) ((_, rastriginSpec), agg) = agg >> do
    model <- sample rastriginSpec
    let zipped = hzip
          (grad (rastrigin @a @dtype @device @tensors @parameters model a)
                (flattenParameters model)
          )
          (gradientsRastrigin @gradients' model a)
    hfoldrM @IO GradientsTestInner () zipped

data LinearForward = LinearForward

instance Apply' LinearForward (Linear inputFeatures outputFeatures dtype device, Tensor device dtype '[batchSize, inputFeatures]) (Tensor device dtype '[batchSize, outputFeatures]) where
  apply' _ (model, input) = forward model input

spec :: Spec
spec = describe "grad" $ do
  it "works if everything has identical device and dtype" $ do
    hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $ hattach
      (Proxy @'( 'CPU, 0), Proxy @'Float)
      (  RastriginSpec @1 @'[2] @'[ 'Float] @'[ '( 'CPU, 0)]
      :. RastriginSpec @2 @'[2, 3] @'[ 'Float, 'Float] @'[ '( 'CPU, 0), '( 'CPU, 0)]
      :. HNil
      )
  it "works if model and loss have different dtypes but live on the same device" $ do
    hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $ hproduct
      (  (Proxy @'( 'CPU, 0), Proxy @'Double)
      :. (Proxy @'( 'CPU, 0), Proxy @'Float)
      :. HNil
      )
      (  RastriginSpec @2 @'[0, 1] @'[ 'Float, 'Double] @'[ '( 'CPU, 0), '( 'CPU, 0)]
      :. RastriginSpec @4 @'[2, 3, 1, 13] @'[ 'Float, 'Double, 'Float, 'Double] @'[ '( 'CPU, 0), '( 'CPU, 0), '( 'CPU, 0), '( 'CPU, 0)]
      :. HNil
      )
  when (  elem (Device { deviceType = CPU,  deviceIndex = 0 }) availableDevices
       && elem (Device { deviceType = CUDA, deviceIndex = 0 }) availableDevices
       ) $ do
    it "works if individual model layers and loss have different dtypes and live on different devices" $ do
      hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $ hproduct
        (  (Proxy @'( 'CPU, 0), Proxy @'Double)
        :. (Proxy @'( 'CUDA, 0), Proxy @'Double)
        :. HNil
        )
        (  RastriginSpec @2 @'[1, 5] @'[ 'Float, 'Double] @'[ '( 'CUDA, 0), '( 'CPU, 0)]
        :. RastriginSpec @4 @'[2, 3, 1, 13] @'[ 'Float, 'Double, 'Float, 'Double] @'[ '( 'CPU, 0), '( 'CUDA, 0), '( 'CPU, 0), '( 'CUDA, 0)]
        :. HNil
        )
    it "works in a data-parallel setting" $ do
      let spec = LinearSpec @10 @5 @'Float @'( 'CPU, 0)
      model <- sample spec
      input <- randn @'[20,10] @'Float @'( 'CPU, 0)
      output <- forwardConcurrently' @'[ '( 'CPU, 0), '( 'CUDA, 0)] @'( 'CPU, 0) model input
      let loss = mseLoss @ReduceMean output zeros
          gradientWeight :. gradientBias :. HNil = grad loss (flattenParameters model)
          output' = forward model input
          loss' = mseLoss @ReduceMean output' zeros
          gradientWeight' :. gradientBias' :. HNil = grad loss' (flattenParameters model)
      (toInt . all) (isclose 1e-08 1e-05 False gradientWeight gradientWeight') `shouldBe` 1
      (toInt . all) (isclose 1e-08 1e-05 False gradientBias gradientBias') `shouldBe` 1
