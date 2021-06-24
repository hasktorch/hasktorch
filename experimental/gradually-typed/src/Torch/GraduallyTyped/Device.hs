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
{-# LANGUAGE QuantifiedConstraints #-}
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
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat (..), Nat, natVal)
import Torch.GraduallyTyped.Prelude (Concat)
import qualified Torch.Internal.Managed.Cast as ATen ()

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

instance (KnownNat deviceId) => KnownDeviceType ('CUDA deviceId) where
  deviceTypeVal = CUDA (fromIntegral . natVal $ Proxy @deviceId)

-- | Data type to represent whether or not the compute device is checked, that is, known to the compiler.
data Device (deviceType :: Type) where
  -- | The compute device is unknown to the compiler.
  UncheckedDevice :: forall deviceType. Device deviceType
  -- | The compute device is known to the compiler.
  Device :: forall deviceType. deviceType -> Device deviceType
  deriving (Show)

data SDevice (deviceType :: Device (DeviceType Nat)) where
  SUncheckedDevice :: DeviceType Int16 -> SDevice 'UncheckedDevice
  SDevice :: forall deviceType. KnownDeviceType deviceType => SDevice ('Device deviceType)

type family DeviceTypeF (device :: Device (DeviceType Nat)) :: DeviceType Nat where
  DeviceTypeF ('Device deviceType) = deviceType

sDeviceType :: forall device deviceType. SDevice device -> DeviceType Int16
sDeviceType (SUncheckedDevice deviceType) = deviceType
sDeviceType SDevice = deviceTypeVal @(DeviceTypeF device)

class KnownDevice (device :: Device (DeviceType Nat)) where
  deviceVal :: Device (DeviceType Int16)

instance KnownDevice 'UncheckedDevice where
  deviceVal = UncheckedDevice

instance (KnownDeviceType deviceType) => KnownDevice ('Device deviceType) where
  deviceVal = Device (deviceTypeVal @deviceType)

class
  DeviceConstraint device (GetDevices f) =>
  WithDeviceC (device :: Device (DeviceType Nat)) (f :: Type)
  where
  type WithDeviceF device f :: Type
  withDevice :: (DeviceType Int16 -> f) -> WithDeviceF device f
  withoutDevice :: WithDeviceF device f -> (DeviceType Int16 -> f)

instance
  DeviceConstraint 'UncheckedDevice (GetDevices f) =>
  WithDeviceC 'UncheckedDevice f
  where
  type WithDeviceF 'UncheckedDevice f = DeviceType Int16 -> f
  withDevice = id
  withoutDevice = id

instance
  ( DeviceConstraint ('Device deviceType) (GetDevices f),
    KnownDeviceType deviceType
  ) =>
  WithDeviceC ('Device deviceType) f
  where
  type WithDeviceF ('Device deviceType) f = f
  withDevice f = f (deviceTypeVal @deviceType)
  withoutDevice = const

type family DeviceConstraint (device :: Device (DeviceType Nat)) (devices :: [Device (DeviceType Nat)]) :: Constraint where
  DeviceConstraint _ '[] = ()
  DeviceConstraint device '[device'] = device ~ device'
  DeviceConstraint _ _ = ()

-- >>> :kind! GetDevices ('Device ('CUDA 0))
-- GetDevices ('Device ('CUDA 0)) :: [Device (DeviceType Nat)]
-- = '[ 'Device ('CUDA 0)]
-- >>> :kind! GetDevices '[ 'Device 'CPU, 'Device ('CUDA 0)]
-- GetDevices '[ 'Device 'CPU, 'Device ('CUDA 0)] :: [Device
--                                                      (DeviceType Nat)]
-- = '[ 'Device 'CPU, 'Device ('CUDA 0)]
-- >>> :kind! GetDevices ('Just ('Device ('CUDA 0)))
-- GetDevices ('Just ('Device ('CUDA 0))) :: [Device (DeviceType Nat)]
-- = '[ 'Device ('CUDA 0)]
type family GetDevices (f :: k) :: [Device (DeviceType Nat)] where
  GetDevices (a :: Device (DeviceType Nat)) = '[a]
  GetDevices (f g) = Concat (GetDevices f) (GetDevices g)
  GetDevices _ = '[]
