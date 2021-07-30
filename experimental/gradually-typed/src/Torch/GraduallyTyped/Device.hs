{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Device where

import Data.Int (Int16)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..), withSomeSing)
import GHC.TypeLits (KnownNat, Nat, SomeNat (..), natVal, someNatVal)
import Torch.GraduallyTyped.Prelude (Concat, IsChecked (..))
import qualified Torch.Internal.Managed.Cast as ATen ()

-- | Data type to represent compute devices.
data DeviceType (deviceId :: Type) where
  -- | The tensor is stored in the CPU's memory.
  CPU :: forall deviceId. DeviceType deviceId
  -- | The tensor is stored the memory of the GPU with ID 'deviceId'.
  CUDA :: forall deviceId. deviceId -> DeviceType deviceId
  deriving (Eq, Ord, Show)

data SDeviceType (deviceType :: DeviceType Nat) where
  SCPU :: SDeviceType 'CPU
  SCUDA :: forall deviceId. KnownNat deviceId => SDeviceType ('CUDA deviceId)

deriving stock instance Show (SDeviceType (deviceType :: DeviceType Nat))

type instance Sing = SDeviceType

instance SingI ('CPU :: DeviceType Nat) where
  sing = SCPU

instance KnownNat deviceId => SingI ('CUDA deviceId) where
  sing = SCUDA @deviceId

type family CUDAF (deviceType :: DeviceType Nat) :: Nat where
  CUDAF ('CUDA deviceId) = deviceId

instance SingKind (DeviceType Nat) where
  type Demote (DeviceType Nat) = DeviceType Int16
  fromSing SCPU = CPU
  fromSing (SCUDA :: Sing deviceId) = CUDA . fromIntegral . natVal $ Proxy @(CUDAF deviceId)
  toSing CPU = SomeSing SCPU
  toSing (CUDA deviceId) = case someNatVal (fromIntegral deviceId) of
    Just (SomeNat (_ :: Proxy deviceId)) -> SomeSing (SCUDA @deviceId)

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
  SDevice :: forall deviceType. SDeviceType deviceType -> SDevice ('Device deviceType)

deriving stock instance Show (SDevice (device :: Device (DeviceType Nat)))

type instance Sing = SDevice

instance SingI deviceType => SingI ('Device (deviceType :: DeviceType Nat)) where
  sing = SDevice $ sing @deviceType

instance SingKind (Device (DeviceType Nat)) where
  type Demote (Device (DeviceType Nat)) = IsChecked (DeviceType Int16)
  fromSing (SUncheckedDevice deviceType) = Unchecked deviceType
  fromSing (SDevice deviceType) = Checked . fromSing $ deviceType
  toSing (Unchecked deviceType) = SomeSing . SUncheckedDevice $ deviceType
  toSing (Checked deviceType) = withSomeSing deviceType $ SomeSing . SDevice

class KnownDevice (device :: Device (DeviceType Nat)) where
  deviceVal :: Device (DeviceType Int16)

instance KnownDevice 'UncheckedDevice where
  deviceVal = UncheckedDevice

instance (KnownDeviceType deviceType) => KnownDevice ('Device deviceType) where
  deviceVal = Device (deviceTypeVal @deviceType)

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
type GetDevices :: k -> [Device (DeviceType Nat)]
type family GetDevices f where
  GetDevices (a :: Device (DeviceType Nat)) = '[a]
  GetDevices (f g) = Concat (GetDevices f) (GetDevices g)
  GetDevices _ = '[]
