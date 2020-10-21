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

-- | Data type to represent compute devices.
data DeviceType (deviceId :: Type) where
  -- | The tensor is stored in the CPU's memory.
  CPU :: forall deviceId. DeviceType deviceId
  -- | The tensor is stored the memory of the GPU with ID 'deviceId'.
  CUDA :: forall deviceId. deviceId -> DeviceType deviceId
  deriving (Show)

class KnownDeviceType (deviceType :: DeviceType Nat) where
  deviceTypeVal :: DeviceType Int16

instance KnownDeviceType 'CPU where
  deviceTypeVal = CPU

instance (KnownNat deviceId) => KnownDeviceType ( 'CUDA deviceId) where
  deviceTypeVal = CUDA (fromIntegral . natVal $ Proxy @deviceId)

-- | Data type to represent whether or not the compute device is checked, that is, known to the compiler.
data Device (deviceType :: Type) where
  -- | The compute device is unknown to the compiler.
  UncheckedDevice :: forall deviceType. Device deviceType
  -- | The compute device is known to the compiler.
  Device :: forall deviceType. deviceType -> Device deviceType
  deriving (Show)

class KnownDevice (device :: Device (DeviceType Nat)) where
  deviceVal :: Device (DeviceType Int16)

instance KnownDevice 'UncheckedDevice where
  deviceVal = UncheckedDevice

instance (KnownDeviceType deviceType) => KnownDevice ( 'Device deviceType) where
  deviceVal = Device (deviceTypeVal @deviceType)

class WithDeviceC (device :: Device (DeviceType Nat)) (f :: Type) where
  type WithDeviceF device f :: Type
  withDevice :: (DeviceType Int16 -> f) -> WithDeviceF device f
  withoutDevice :: WithDeviceF device f -> (DeviceType Int16 -> f)

instance WithDeviceC 'UncheckedDevice f where
  type WithDeviceF 'UncheckedDevice f = DeviceType Int16 -> f
  withDevice = id
  withoutDevice = id

instance (KnownDeviceType deviceType) => WithDeviceC ( 'Device deviceType) f where
  type WithDeviceF ( 'Device deviceType) f = f
  withDevice f = f (deviceTypeVal @deviceType)
  withoutDevice = const

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