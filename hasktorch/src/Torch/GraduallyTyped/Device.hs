{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Device where

import Data.Int (Int16)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits
  ( KnownNat,
    Nat,
    natVal,
  )
import qualified Torch.Internal.Managed.Cast as ATen ()
import Type.Errors.Pretty (TypeError, type (%), type (<>))

data DeviceType (deviceId :: Type) where
  CPU :: forall deviceId. DeviceType deviceId
  CUDA :: forall deviceId. deviceId -> DeviceType deviceId
  deriving (Show)

data Device (deviceType :: Type) where
  UncheckedDevice :: forall deviceType. Device deviceType
  Device :: forall deviceType. deviceType -> Device deviceType
  deriving (Show)

class KnownDevice (device :: Device (DeviceType Nat)) where
  deviceVal :: Device (DeviceType Int16)

instance KnownDevice 'UncheckedDevice where
  deviceVal = UncheckedDevice

instance KnownDevice ( 'Device 'CPU) where
  deviceVal = Device CPU

instance (KnownNat deviceId) => KnownDevice ( 'Device ( 'CUDA deviceId)) where
  deviceVal = Device (CUDA (fromIntegral . natVal $ Proxy @deviceId))

class WithDeviceC (isAnyDevice :: Bool) (device :: Device (DeviceType Nat)) (f :: Type) where
  type WithDeviceF isAnyDevice f :: Type
  withDevice :: (DeviceType Int16 -> f) -> WithDeviceF isAnyDevice f

instance WithDeviceC 'True device f where
  type WithDeviceF 'True f = DeviceType Int16 -> f
  withDevice = id

instance (KnownDevice device) => WithDeviceC 'False device f where
  type WithDeviceF 'False f = f
  withDevice f = case deviceVal @device of Device device -> f device

type family UnifyDeviceF (device :: Device (DeviceType Nat)) (device' :: Device (DeviceType Nat)) :: Device (DeviceType Nat) where
  UnifyDeviceF 'UncheckedDevice _ = 'UncheckedDevice
  UnifyDeviceF _ 'UncheckedDevice = 'UncheckedDevice
  UnifyDeviceF ( 'Device deviceType) ( 'Device deviceType) = 'Device deviceType
  UnifyDeviceF ( 'Device deviceType) ( 'Device deviceType') =
    TypeError
      ( "The supplied tensors must be on the same device, "
          % "but different device locations were found:"
          % ""
          % "    " <> deviceType <> " and " <> deviceType' <> "."
          % ""
      )