{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module Torch.Typed.AutogradSpec
  ( Torch.Typed.AutogradSpec.spec,
  )
where

import Control.Monad (when)
import Data.Kind
import Data.Maybe ()
import Data.Proxy
import Data.Reflection ()
import GHC.Exts (toList)
import GHC.Generics
import GHC.TypeLits
import System.IO.Unsafe ()
import Test.Hspec (Spec, describe, it, shouldBe)
import Test.QuickCheck ()
import Torch (ATenTensor)
import Torch.Internal.Class (Castable)
import Torch.Typed
import Torch.Typed.AuxiliarySpec
import Prelude hiding
  ( all,
    cos,
    sin,
  )

data
  RastriginLayerSpec
    (n :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  = RastriginLayerSpec
  deriving (Show, Eq)

data
  RastriginLayer
    (n :: Nat)
    (dtype :: DType)
    (device :: (DeviceType, Nat))
  where
  RastriginLayer ::
    forall n dtype device.
    {x :: Parameter device dtype '[n]} ->
    RastriginLayer n dtype device
  deriving (Show, Generic, Parameterized)

instance
  ( RandDTypeIsValid device dtype,
    KnownNat n,
    KnownDType dtype,
    KnownDevice device
  ) =>
  Randomizable
    (RastriginLayerSpec n dtype device)
    (RastriginLayer n dtype device)
  where
  sample _ = RastriginLayer <$> (makeIndependent =<< randn)

rastriginLayer' ::
  forall device dtype a n shape.
  ( SumDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype,
    All Scalar '[a, n],
    KnownDType (SumDType dtype),
    KnownDevice device
  ) =>
  Tensor device dtype shape ->
  a ->
  n ->
  Tensor device (SumDType dtype) '[]
rastriginLayer' x a n =
  (mulScalar a . mulScalar n $ ones)
    + sumAll (x * x - (mulScalar a . cos . mulScalar (2 * pi :: Double)) x)

gradientsRastriginLayer' ::
  forall device dtype a shape.
  (KnownDevice device, StandardFloatingPointDTypeValidation device dtype, Scalar a) =>
  Tensor device dtype shape ->
  a ->
  Tensor device dtype shape
gradientsRastriginLayer' x a =
  mulScalar
    (2 :: Int)
    ( x
        + ( mulScalar a . mulScalar (pi :: Double)
              . sin
              . mulScalar (2 * pi :: Double)
          )
          x
    )

data
  RastriginStackSpec
    (num :: Nat)
    (ns :: [Nat])
    (dtypes :: [DType])
    (devices :: [(DeviceType, Nat)])
  = RastriginStackSpec
  deriving (Show, Eq)

data
  RastriginStack
    (num :: Nat)
    (ns :: [Nat])
    (dtypes :: [DType])
    (devices :: [(DeviceType, Nat)])
  where
  Rastrigin1 ::
    forall n dtype device.
    RastriginLayer n dtype device ->
    RastriginStack 1 '[n] '[dtype] '[device]
  RastriginK ::
    forall num n ns dtype dtypes device devices.
    RastriginLayer n dtype device ->
    RastriginStack num ns dtypes devices ->
    RastriginStack (num + 1) (n ': ns) (dtype ': dtypes) (device ': devices)

deriving instance Show (RastriginStack num ns dtypes devices)

class RastriginStackParameterized (flag :: Bool) num ns dtypes devices where
  type RastriginStackParameters flag num ns dtypes devices :: [Type]
  rastriginStackFlattenParameters ::
    Proxy flag ->
    RastriginStack num ns dtypes devices ->
    HList (RastriginStackParameters flag num ns dtypes devices)
  rastriginStackReplaceParameters ::
    Proxy flag ->
    RastriginStack num ns dtypes devices ->
    HList (RastriginStackParameters flag num ns dtypes devices) ->
    RastriginStack num ns dtypes devices

instance
  Parameterized (RastriginLayer n dtype device) =>
  RastriginStackParameterized 'False 1 '[n] '[dtype] '[device]
  where
  type
    RastriginStackParameters 'False 1 '[n] '[dtype] '[device] =
      Parameters (RastriginLayer n dtype device)
  rastriginStackFlattenParameters _ (Rastrigin1 rastriginLayer) = flattenParameters rastriginLayer
  rastriginStackReplaceParameters _ (Rastrigin1 rastriginLayer) parameters =
    Rastrigin1 $ replaceParameters rastriginLayer parameters

instance
  ( Parameterized (RastriginLayer n dtype device),
    Parameterized (RastriginStack (num - 1) ns dtypes devices),
    HAppendFD
      (Parameters (RastriginLayer n dtype device))
      (Parameters (RastriginStack (num - 1) ns dtypes devices))
      ( Parameters (RastriginLayer n dtype device)
          ++ Parameters (RastriginStack (num - 1) ns dtypes devices)
      ),
    1 <= num,
    numM1 ~ num - 1,
    0 <= numM1
  ) =>
  RastriginStackParameterized 'True num (n ': ns) (dtype ': dtypes) (device ': devices)
  where
  type
    RastriginStackParameters 'True num (n ': ns) (dtype ': dtypes) (device ': devices) =
      (Parameters (RastriginLayer n dtype device) ++ Parameters (RastriginStack (num - 1) ns dtypes devices))
  rastriginStackFlattenParameters _ (RastriginK rastriginLayer rastriginStack) =
    let parameters = flattenParameters rastriginLayer
        parameters' = flattenParameters @(RastriginStack numM1 ns dtypes devices) rastriginStack
     in parameters `happendFD` parameters'
  rastriginStackReplaceParameters _ (RastriginK rastriginLayer rastriginStack) parameters'' =
    let (parameters, parameters') = hunappendFD parameters''
        rastriginLayer' = replaceParameters rastriginLayer parameters
        rastriginStack' =
          replaceParameters @(RastriginStack (num - 1) ns dtypes devices)
            rastriginStack
            parameters'
     in RastriginK rastriginLayer' rastriginStack'

instance
  ( 1 <= num,
    (2 <=? num) ~ flag,
    RastriginStackParameterized flag num ns dtypes devices
  ) =>
  Parameterized (RastriginStack num ns dtypes devices)
  where
  type
    Parameters (RastriginStack num ns dtypes devices) =
      RastriginStackParameters (2 <=? num) num ns dtypes devices
  flattenParameters = rastriginStackFlattenParameters (Proxy :: Proxy flag)
  replaceParameters = rastriginStackReplaceParameters (Proxy :: Proxy flag)

class RastriginStackRandomizable (flag :: Bool) num ns dtypes devices where
  rastriginStackSample ::
    Proxy flag ->
    RastriginStackSpec num ns dtypes devices ->
    IO (RastriginStack num ns dtypes devices)

instance
  ( KnownNat n,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  RastriginStackRandomizable 'False 1 '[n] '[dtype] '[device]
  where
  rastriginStackSample _ _ = Rastrigin1 <$> (sample $ RastriginLayerSpec @n @dtype @device)

instance
  ( KnownNat n,
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype,
    Randomizable
      (RastriginStackSpec (num - 1) ns dtypes devices)
      (RastriginStack (num - 1) ns dtypes devices),
    1 <= num
  ) =>
  RastriginStackRandomizable 'True num (n ': ns) (dtype ': dtypes) (device ': devices)
  where
  rastriginStackSample _ _ =
    RastriginK
      <$> (sample $ RastriginLayerSpec @n @dtype @device)
      <*> ( sample
              @(RastriginStackSpec (num - 1) ns dtypes devices)
              @(RastriginStack (num - 1) ns dtypes devices)
              $ RastriginStackSpec
          )

instance
  ( 1 <= num,
    (2 <=? num) ~ flag,
    RandDTypeIsValid device dtype,
    KnownDType dtype,
    KnownDevice device,
    RastriginStackRandomizable flag num (n ': ns) (dtype ': dtypes) (device ': devices)
  ) =>
  Randomizable
    (RastriginStackSpec num (n ': ns) (dtype ': dtypes) (device ': devices))
    (RastriginStack num (n ': ns) (dtype ': dtypes) (device ': devices))
  where
  sample = rastriginStackSample (Proxy :: Proxy flag)

data
  RastriginSpec
    (num :: Nat)
    (ns :: [Nat])
    (dtypes :: [DType])
    (devices :: [(DeviceType, Nat)])
  = RastriginSpec
  deriving (Show, Eq)

data
  Rastrigin
    (num :: Nat)
    (ns :: [Nat])
    (dtypes :: [DType])
    (devices :: [(DeviceType, Nat)]) = Rastrigin
  { rastriginStack :: RastriginStack num ns dtypes devices
  }
  deriving (Show, Generic)

deriving instance
  ( 1 <= num,
    Parameterized (RastriginStack num ns dtypes devices)
  ) =>
  Parameterized (Rastrigin num ns dtypes devices)

instance
  ( Randomizable
      (RastriginStackSpec num ns dtypes devices)
      (RastriginStack num ns dtypes devices)
  ) =>
  Randomizable
    (RastriginSpec num ns dtypes devices)
    (Rastrigin num ns dtypes devices)
  where
  sample _ =
    Rastrigin <$> (sample $ RastriginStackSpec @num @ns @dtypes @devices)

data RastriginA a dv dt = RastriginA a dv dt

instance
  ( Scalar a,
    KnownNat n,
    All KnownDType [SumDType dtype, dtype'],
    All KnownDevice [device, device'],
    SumDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype
  ) =>
  Apply'
    (RastriginA a (Proxy device') (Proxy dtype'))
    (Parameter device dtype '[n])
    (Tensor device' dtype' '[])
  where
  apply' (RastriginA a _ _) parameter =
    toDevice @device'
      . toDType @dtype' @(SumDType dtype)
      . rastriginLayer' (toDependent parameter) a
      $ natValI @n

rastrigin ::
  forall a dtype device tensors parameters num ns dtypes devices shape.
  ( SumDType dtype ~ dtype,
    SumDTypeIsValid device dtype,
    parameters ~ Parameters (Rastrigin num ns dtypes devices),
    Parameterized (Rastrigin num ns dtypes devices),
    HMap' (RastriginA a (Proxy device) (Proxy dtype)) parameters tensors,
    Castable (HList tensors) [ATenTensor],
    '(shape, dtype, device) ~ Stack 0 tensors,
    DropValue shape 0 ~ '[]
  ) =>
  Rastrigin num ns dtypes devices ->
  a ->
  Tensor device dtype '[]
rastrigin model a =
  sumDim @0
    . stack @0
    . hmap' (RastriginA a (Proxy @device) (Proxy @dtype))
    . flattenParameters
    $ model

data GradientsRastriginA a = GradientsRastriginA a

instance
  ( KnownDevice device,
    StandardFloatingPointDTypeValidation device dtype,
    Scalar a
  ) =>
  Apply' (GradientsRastriginA a) (Parameter device dtype '[n]) (Tensor device dtype '[n])
  where
  apply' (GradientsRastriginA a) parameter = gradientsRastriginLayer' (toDependent parameter) $ a

gradientsRastrigin ::
  forall gradients a num ns dtypes devices parameters.
  ( HMap' (GradientsRastriginA a) parameters gradients,
    parameters ~ Parameters (Rastrigin num ns dtypes devices),
    Parameterized (Rastrigin num ns dtypes devices)
  ) =>
  Rastrigin num ns dtypes devices ->
  a ->
  HList gradients
gradientsRastrigin model a =
  hmap'
    (GradientsRastriginA a)
    . flattenParameters
    $ model

data GradientsTestInner = GradientsTestInner

instance
  ( TensorOptions shape dtype device
  ) =>
  Apply'
    GradientsTestInner
    ((Tensor device dtype shape, Tensor device dtype shape), IO ())
    (IO ())
  where
  apply' _ ((a, b), agg) =
    agg >> do
      checkDynamicTensorAttributes a
      checkDynamicTensorAttributes b
      (toList . Just . toDevice @'( 'CPU, 0) @device . unsqueeze @0 . all)
        (isclose 1e-05 1e-08 False a b)
        `shouldBe` [True]

data GradientsTestOuter a = GradientsTestOuter a

instance
  ( Randomizable
      (RastriginStackSpec num ns dtypes devices)
      (RastriginStack num ns dtypes devices),
    HasGrad (HList parameters) (HList gradients),
    SumDType dtype ~ dtype,
    SumDTypeIsValid device dtype,
    parameters ~ Parameters (Rastrigin num ns dtypes devices),
    Parameterized (Rastrigin num ns dtypes devices),
    HMap' (RastriginA a (Proxy device) (Proxy dtype)) parameters tensors,
    Castable (HList tensors) [ATenTensor],
    '(shape, dtype, device) ~ Stack 0 tensors,
    DropValue shape 0 ~ '[],
    HMap' (GradientsRastriginA a) parameters gradients',
    HZip gradients gradients' zs,
    HFoldrM IO GradientsTestInner () zs ()
  ) =>
  Apply'
    (GradientsTestOuter a)
    ( ( (Proxy device, Proxy dtype),
        RastriginSpec num ns dtypes devices
      ),
      IO ()
    )
    (IO ())
  where
  apply' (GradientsTestOuter a) ((_, rastriginSpec), agg) =
    agg >> do
      model <- sample rastriginSpec
      let zipped =
            hzip
              ( grad
                  (rastrigin @a @dtype @device @tensors @parameters model a)
                  (flattenParameters model)
              )
              (gradientsRastrigin @gradients' model a)
      hfoldrM @IO GradientsTestInner () zipped

data LinearForward = LinearForward

instance
  Apply'
    LinearForward
    ( Linear inputFeatures outputFeatures dtype device,
      Tensor device dtype '[batchSize, inputFeatures]
    )
    (Tensor device dtype '[batchSize, outputFeatures])
  where
  apply' _ (model, input) = forward model input

spec :: Spec
spec = describe "grad" $ do
  it "works if everything has identical device and dtype" $ do
    hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $
      hattach
        (Proxy @'( 'CPU, 0), Proxy @'Float)
        ( RastriginSpec @1 @'[2] @'[ 'Float] @'[ '( 'CPU, 0)]
            :. RastriginSpec @2 @'[2, 3] @'[ 'Float, 'Float] @'[ '( 'CPU, 0), '( 'CPU, 0)]
            :. HNil
        )
  it "works if model and loss have different dtypes but live on the same device" $ do
    hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $
      hproduct
        ( (Proxy @'( 'CPU, 0), Proxy @'Double)
            :. (Proxy @'( 'CPU, 0), Proxy @'Float)
            :. HNil
        )
        ( RastriginSpec @2 @'[0, 1] @'[ 'Float, 'Double] @'[ '( 'CPU, 0), '( 'CPU, 0)]
            :. RastriginSpec @4 @'[2, 3, 1, 13] @'[ 'Float, 'Double, 'Float, 'Double] @'[ '( 'CPU, 0), '( 'CPU, 0), '( 'CPU, 0), '( 'CPU, 0)]
            :. HNil
        )
  when
    ( elem (Device {deviceType = CPU, deviceIndex = 0}) availableDevices
        && elem (Device {deviceType = CUDA, deviceIndex = 0}) availableDevices
    )
    $ do
      it "works if individual model layers and loss have different dtypes and live on different devices" $ do
        hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $
          hproduct
            ( (Proxy @'( 'CPU, 0), Proxy @'Double)
                :. (Proxy @'( 'CUDA, 0), Proxy @'Double)
                :. HNil
            )
            ( RastriginSpec @2 @'[1, 5] @'[ 'Float, 'Double] @'[ '( 'CUDA, 0), '( 'CPU, 0)]
                :. RastriginSpec @4 @'[2, 3, 1, 13] @'[ 'Float, 'Double, 'Float, 'Double] @'[ '( 'CPU, 0), '( 'CUDA, 0), '( 'CPU, 0), '( 'CUDA, 0)]
                :. HNil
            )
-- ToDo: The error in the lower digits is very large as if using half precision.
--       it "works in a data-parallel setting" $ do
--         let spec = LinearSpec @10 @5 @'Float @'( 'CPU, 0)
--         model <- sample spec
--         input <- randn @'[20, 10] @'Float @'( 'CPU, 0)
--         output <- forwardConcurrently' @'[ '( 'CPU, 0), '( 'CUDA, 0)] @'( 'CPU, 0) model input
--         let loss = mseLoss @ReduceMean output zeros
--             gradientWeight :. gradientBias :. HNil = grad loss (flattenParameters model)
--             output' = forward model input
--             loss' = mseLoss @ReduceMean output' zeros
--             gradientWeight' :. gradientBias' :. HNil = grad loss' (flattenParameters model)
--         print (gradientWeight,gradientWeight')
-- 
--        (toInt . all) (isclose 1e-08 1e-05 False gradientWeight gradientWeight') `shouldBe` 1
--        (toInt . all) (isclose 1e-08 1e-05 False gradientBias gradientBias') `shouldBe` 1
