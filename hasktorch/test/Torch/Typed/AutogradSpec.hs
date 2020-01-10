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
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE StandaloneDeriving #-}

module Torch.Typed.AutogradSpec
  ( Torch.Typed.AutogradSpec.spec
  )
where

import           Prelude                 hiding ( all
                                                , cos
                                                , sin
                                                , sum
                                                )
import           Control.Monad                  ( when )
import           Control.Exception.Safe
import           Foreign.Storable
import           Torch.HList
import           Data.Kind
import           Data.Proxy
import           Data.Maybe
import           Data.Reflection
import           GHC.Generics
import           GHC.TypeLits
import           GHC.Exts
import           System.IO.Unsafe

import           Test.Hspec
import           Test.QuickCheck

import qualified Torch.Internal.Cast                     as ATen
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Type                     as ATen
import qualified Torch.Internal.Managed.Type.Tensor      as ATen
import qualified Torch.Internal.Managed.Type.Context     as ATen
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functional              as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.Scalar                  as D
import qualified Torch.NN                      as A
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Functional
import           Torch.Typed.Factories
import           Torch.Typed.NN
import           Torch.Typed.Autograd
import           Torch.Typed.Optim
import           Torch.Typed.Serialize
import           Torch.Typed.AuxSpec

data RastriginLayerSpec (n :: Nat)
                        (dtype :: D.DType)
                        (device :: (D.DeviceType, Nat))
  = RastriginLayerSpec deriving (Show, Eq)

data RastriginLayer (n :: Nat)
                    (dtype :: D.DType)
                    (device :: (D.DeviceType, Nat))
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
  ) => A.Randomizable (RastriginLayerSpec n dtype device)
                      (RastriginLayer     n dtype device)
 where
  sample _ = RastriginLayer <$> (makeIndependent =<< randn)

rastriginLayer'
  :: forall device dtype a n shape
   . ( SumDTypeIsValid device dtype
     , StandardFloatingPointDTypeValidation device dtype
     , D.Scalar a
     , D.Scalar n
     , KnownDType (SumDType dtype)
     , KnownDevice device
     )
  => Tensor device dtype shape
  -> a
  -> n
  -> Tensor device (SumDType dtype) '[]
rastriginLayer' x a n = (cmul a . cmul n $ ones)
  + sumAll (x * x - (cmul a . cos . cmul (2 * pi :: Double)) x)

gradientsRastriginLayer'
  :: forall device dtype a shape
   . (StandardFloatingPointDTypeValidation device dtype, D.Scalar a)
  => Tensor device dtype shape
  -> a
  -> Tensor device dtype shape
gradientsRastriginLayer' x a = cmul
  (2 :: Int)
  (x + (cmul a . cmul (pi :: Double) . sin . cmul (2 * pi :: Double)) x)

data RastriginStackSpec (num :: Nat)
                        (ns :: [Nat])
                        (dtypes :: [D.DType])
                        (devices :: [(D.DeviceType, Nat)])
  = RastriginStackSpec deriving (Show, Eq)

data RastriginStack (num :: Nat)
                    (ns :: [Nat])
                    (dtypes :: [D.DType])
                    (devices :: [(D.DeviceType, Nat)])
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
  ( GParameterized (K1 R (RastriginLayer n dtype device)) parameters
  ) => GParameterized (K1 R (RastriginStack 1 '[n] '[dtype] '[device])) parameters where
  gFlattenParameters (K1 (Rastrigin1 rastrigin))
    = gFlattenParameters (K1 @R rastrigin)
  gReplaceParameters (K1 (Rastrigin1 rastrigin)) parameters
    = K1 (Rastrigin1 (unK1 (gReplaceParameters (K1 @R rastrigin) parameters)))

instance {-# OVERLAPPABLE #-}
  ( 2 <= num
  , GParameterized (K1 R (RastriginStack (num - 1) ns dtypes devices)) parameters
  , GParameterized (K1 R (RastriginLayer n dtype device)) parameters'
  , HAppendFD parameters parameters' parameters''
  , parameters'' ~ (parameters ++ parameters')
  ) => GParameterized (K1 R (RastriginStack num (n ': ns) (dtype ': dtypes) (device ': devices))) parameters'' where
  gFlattenParameters (K1 (RastriginK rastriginStack rastriginLayer))
    = let parameters  = gFlattenParameters (K1 @R rastriginStack)
          parameters' = gFlattenParameters (K1 @R rastriginLayer)
      in  parameters `happendFD` parameters'
  gReplaceParameters (K1 (RastriginK rastriginStack rastriginLayer)) parameters''
    = let (parameters, parameters') = hunappendFD parameters''
          rastriginStack'           = unK1 (gReplaceParameters (K1 @R rastriginStack) parameters)
          rastriginLayer'           = unK1 (gReplaceParameters (K1 @R rastriginLayer) parameters')
      in  K1 (RastriginK rastriginStack' rastriginLayer')

instance {-# OVERLAPS #-}
  ( RandDTypeIsValid device dtype
  , KnownNat n
  , KnownDType dtype
  , KnownDevice device
  ) => A.Randomizable (RastriginStackSpec 1 '[n] '[dtype] '[device])
                      (RastriginStack     1 '[n] '[dtype] '[device])
 where
  sample _ = Rastrigin1 <$> (A.sample $ RastriginLayerSpec @n @dtype @device)

instance {-# OVERLAPPABLE #-}
  ( 2 <= num
  , RandDTypeIsValid device dtype
  , KnownNat n
  , KnownDType dtype
  , KnownDevice device
  , A.Randomizable (RastriginStackSpec (num - 1) ns dtypes devices)
                   (RastriginStack     (num - 1) ns dtypes devices)
  ) => A.Randomizable (RastriginStackSpec num (n ': ns) (dtype ': dtypes) (device ': devices))
                      (RastriginStack     num (n ': ns) (dtype ': dtypes) (device ': devices))
 where
  sample _ =
    RastriginK
      <$> (A.sample $ RastriginStackSpec @(num - 1) @ns @dtypes @devices)
      <*> (A.sample $ RastriginLayerSpec @n @dtype @device)

data RastriginSpec (num :: Nat)
                   (ns :: [Nat])
                   (dtypes :: [D.DType])
                   (devices :: [(D.DeviceType, Nat)])
  = RastriginSpec deriving (Show, Eq)

data Rastrigin (num :: Nat)
               (ns :: [Nat])
               (dtypes :: [D.DType])
               (devices :: [(D.DeviceType, Nat)])
  = Rastrigin
      { rastriginStack :: RastriginStack num ns dtypes devices
      }
  deriving (Show, Generic)

instance
  ( A.Randomizable (RastriginStackSpec num ns dtypes devices)
                   (RastriginStack     num ns dtypes devices)
  ) => A.Randomizable (RastriginSpec num ns dtypes devices)
                      (Rastrigin     num ns dtypes devices)
 where
  sample _ =
    Rastrigin <$> (A.sample $ RastriginStackSpec @num @ns @dtypes @devices)

data RastriginA a dv dt = RastriginA a dv dt

instance
  ( D.Scalar a
  , KnownNat n
  , KnownDType (SumDType dtype)
  , KnownDType dtype'
  , KnownDevice device
  , KnownDevice device'
  , SumDTypeIsValid device dtype
  , StandardFloatingPointDTypeValidation device dtype
  ) => Apply' (RastriginA a (Proxy device') (Proxy dtype')) (Parameter device dtype '[n]) (Tensor device' dtype' '[]) where
  apply' (RastriginA a _ _) parameter =
    Torch.Typed.Tensor.toDevice @device' . Torch.Typed.Tensor.toDType @dtype' . rastriginLayer' (toDependent parameter) a $ natValI @n

rastrigin
  :: forall a dtype device tensors parameters num ns dtypes devices shape
   . ( SumDType dtype ~ dtype
     , SumDTypeIsValid device dtype
     , Parameterized (Rastrigin num ns dtypes devices) parameters
     , HMap' (RastriginA a (Proxy device) (Proxy dtype)) parameters tensors
     , ATen.Castable (HList tensors) [D.ATenTensor]
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
  ( StandardFloatingPointDTypeValidation device dtype
  , D.Scalar a
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
  ) => Apply GradientsTestInner (Tensor device dtype shape, Tensor device dtype shape) (() -> IO ()) where
  apply _ (a, b) _ = do
    checkDynamicTensorAttributes a
    checkDynamicTensorAttributes b
    (toList . Just . Torch.Typed.Tensor.toDevice @'( 'D.CPU, 0) . unsqueeze @0 . all)
        (isclose 1e-05 1e-08 False a b)
      `shouldBe` [True]

data GradientsTestOuter a = GradientsTestOuter a

instance
  ( A.Randomizable (RastriginStackSpec num ns dtypes devices) (RastriginStack num ns dtypes devices)
  , HasGrad (HList parameters) (HList gradients)
  , SumDType dtype ~ dtype
  , SumDTypeIsValid device dtype
  , Parameterized (Rastrigin num ns dtypes devices) parameters
  , HMap' (RastriginA a (Proxy device) (Proxy dtype)) parameters tensors
  , ATen.Castable (HList tensors) [D.ATenTensor]
  , '(shape, dtype, device) ~ Stack 0 tensors
  , DropValue shape 0 ~ '[]
  , HMap' (GradientsRastriginA a) parameters gradients'
  , HZip gradients gradients' zs
  , HFoldrM IO GradientsTestInner () zs
  ) => Apply (GradientsTestOuter a) ((Proxy device, Proxy dtype), RastriginSpec num ns dtypes devices) (() -> IO ()) where
  apply (GradientsTestOuter a) (_, rastriginSpec) _ = do
    model <- A.sample rastriginSpec
    let zipped = hzip
          (grad (rastrigin @a @dtype @device @tensors @parameters model a)
                (flattenParameters model)
          )
          (gradientsRastrigin @gradients' model a)
    hfoldrM @IO GradientsTestInner () zipped

spec :: Spec
spec = describe "grad" $ do
  it "works if everything has identical device and dtype" $ do
    hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $ hattach
      (Proxy @'( 'D.CPU, 0), Proxy @'D.Float)
      (  RastriginSpec @1 @'[2] @'[ 'D.Float] @'[ '( 'D.CPU, 0)]
      :. RastriginSpec @2 @'[2, 3] @'[ 'D.Float, 'D.Float] @'[ '( 'D.CPU, 0), '( 'D.CPU, 0)]
      :. HNil
      )
  it "works if model and loss have different dtypes but live on the same device" $ do
    hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $ hproduct
      (  (Proxy @'( 'D.CPU, 0), Proxy @'D.Double)
      :. (Proxy @'( 'D.CPU, 0), Proxy @'D.Float)
      :. HNil
      )
      (  RastriginSpec @2 @'[0, 1] @'[ 'D.Float, 'D.Double] @'[ '( 'D.CPU, 0), '( 'D.CPU, 0)]
      :. RastriginSpec @4 @'[2, 3, 1, 13] @'[ 'D.Float, 'D.Double, 'D.Float, 'D.Double] @'[ '( 'D.CPU, 0), '( 'D.CPU, 0), '( 'D.CPU, 0), '( 'D.CPU, 0)]
      :. HNil
      )
  when (  elem (D.Device { D.deviceType = D.CPU,  D.deviceIndex = 0 }) availableDevices
       && elem (D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 }) availableDevices
       ) $ do
    it "works if individual model layers and loss have different dtypes and live on different devices" $ do
      hfoldrM @IO (GradientsTestOuter (10 :: Int)) () $ hproduct
        (  (Proxy @'( 'D.CPU, 0), Proxy @'D.Double)
        :. (Proxy @'( 'D.CUDA, 0), Proxy @'D.Double)
        :. HNil
        )
        (  RastriginSpec @2 @'[1, 5] @'[ 'D.Float, 'D.Double] @'[ '( 'D.CUDA, 0), '( 'D.CPU, 0)]
        :. RastriginSpec @4 @'[2, 3, 1, 13] @'[ 'D.Float, 'D.Double, 'D.Float, 'D.Double] @'[ '( 'D.CPU, 0), '( 'D.CUDA, 0), '( 'D.CPU, 0), '( 'D.CUDA, 0)]
        :. HNil
        )
