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
{-# LANGUAGE FunctionalDependencies #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.AuxSpec where

import           Prelude                 hiding ( sin )
import           Control.Exception.Safe
import           Foreign.Storable
import           Torch.HList
import           Data.Proxy
import           Data.Reflection
import           GHC.TypeLits
import           System.IO.Unsafe

import           Test.Hspec
import           Test.QuickCheck

import           Torch.Internal.Cast
import qualified Torch.Internal.Managed.Type.Context     as ATen
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functional               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor

instance Semigroup Spec where
  (<>) a b = a >> b

instance Monoid Spec where
  mempty = pure ()

checkDynamicTensorAttributes
  :: forall device dtype shape
   . (TensorOptions shape dtype device)
  => Tensor device dtype shape
  -> IO ()
checkDynamicTensorAttributes t = do
  D.device untyped `shouldBe` optionsRuntimeDevice @shape @dtype @device
  D.dtype untyped `shouldBe` optionsRuntimeDType @shape @dtype @device
  D.shape untyped `shouldBe` optionsRuntimeShape @shape @dtype @device
  where untyped = toDynamic t

allFloatingPointDTypes :: _
allFloatingPointDTypes = withHalf standardFloatingPointDTypes

standardFloatingPointDTypes :: _
standardFloatingPointDTypes = Proxy @ 'D.Float :. Proxy @ 'D.Double :. HNil

allDTypes :: _
allDTypes = withHalf almostAllDTypes

withHalf :: _ -> _
withHalf dtypes = Proxy @ 'D.Half :. dtypes

almostAllDTypes :: _
almostAllDTypes = withBool standardDTypes

withBool :: _ -> _
withBool dtypes = Proxy @ 'D.Bool :. dtypes

standardDTypes :: _
standardDTypes =
  Proxy @ 'D.UInt8
    :. Proxy @ 'D.Int8
    :. Proxy @ 'D.Int16
    :. Proxy @ 'D.Int32
    :. Proxy @ 'D.Int64
    :. standardFloatingPointDTypes

cpu :: _
cpu = Proxy @'( 'D.CPU, 0)

cuda0 :: _
cuda0 = Proxy @'( 'D.CUDA, 0)

-- data SomeDevices where
--   SomeDevices :: forall (devices :: [(D.DeviceType, Nat)]) deviceList . (All KnownDevice devices, HasDeviceHList devices deviceList) => Proxy devices -> SomeDevices

-- someDevices :: [D.Device] -> SomeDevices
-- someDevices [] = SomeDevices $ Proxy @'[]
-- someDevices (h : t) = case someDevice h of
--   (SomeDevice (Proxy :: Proxy ht)) -> case someDevices t of
--       (SomeDevices (Proxy :: Proxy tt)) -> SomeDevices $ Proxy @(ht ': tt)

-- class HasDeviceHList (devices :: [(D.DeviceType, Nat)]) deviceList | devices -> deviceList where
--   deviceHList :: HList deviceList

-- instance HasDeviceHList '[] '[] where
--   deviceHList = HNil

-- instance HasDeviceHList ds ds' => HasDeviceHList (d ': ds) (Proxy d ': ds') where
--   deviceHList = Proxy @d :. deviceHList @ds

-- spec' :: [D.Device] -> Spec
-- spec' devices = case someDevices devices of
--   (SomeDevices (Proxy :: Proxy devices)) -> do
--     let devices' = deviceHList @devices
--     return ()

-- someDeviceProxies :: [D.Device] -> SomeDeviceProxies
-- someDeviceProxies [] = SomeDeviceProxies $ Proxy @'[]
-- someDeviceProxies (h : t) = case someDevice h of
--   (SomeDevice proxy@(Proxy :: Proxy ht)) -> case someDeviceProxies t of
--       (SomeDeviceProxies (Proxy :: Proxy tt)) -> SomeDeviceProxies $ Proxy @(proxy ': tt)

availableDevices =
  let hasCuda = unsafePerformIO $ cast0 ATen.hasCUDA
  in  [D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 }]
        <> (if hasCuda
             then [D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 }]
             else mempty
           )

spec :: Spec
spec = return ()
