{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
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
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
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
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..))
import Data.Singletons.Prelude.Check (Check (Unchecked), SCheck (..), type SChecked, type SUnchecked)
import GHC.TypeLits (KnownNat, Nat, SomeNat (..), natVal, someNatVal)
import Torch.GraduallyTyped.Prelude (Concat)
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

type Device :: Type

type Device = Check (DeviceType Nat) Type

type SDevice :: Device -> Type

type SDevice device = SCheck device

type SCheckedDevice :: DeviceType Nat -> Type

type SCheckedDevice deviceType = SChecked deviceType

pattern SCheckedDevice :: forall (a :: DeviceType Nat). Sing a -> SCheckedDevice a
pattern SCheckedDevice deviceType = SChecked deviceType

type SUncheckedDevice :: Type

type SUncheckedDevice = SUnchecked (DeviceType Int16)

pattern SUncheckedDevice :: DeviceType Int16 -> SUncheckedDevice
pattern SUncheckedDevice deviceType = SUnchecked deviceType

foo = SCheckedDevice (SCUDA @1)

bar = SUncheckedDevice (CUDA 0)

baz = case fromSing bar of
  Unchecked b -> b

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
type GetDevices :: k -> [Device]
type family GetDevices f where
  GetDevices (a :: Device) = '[a]
  GetDevices (f g) = Concat (GetDevices f) (GetDevices g)
  GetDevices _ = '[]
