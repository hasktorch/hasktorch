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
import Foreign.ForeignPtr (ForeignPtr)
import GHC.TypeLits
  ( KnownNat,
    KnownSymbol,
    Nat,
    Symbol,
    natVal,
    symbolVal,
  )
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.Internal.Cast (cast0, cast1)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Autograd as ATen
import qualified Torch.Internal.Managed.Cast as ATen ()
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Type as ATen (Tensor, TensorList)

data DeviceType (deviceId :: Type) where
  CPU :: forall deviceId. DeviceType deviceId
  CUDA :: forall deviceId. deviceId -> DeviceType deviceId
  deriving Show

data Device (deviceType :: Type) where
  AnyDevice :: forall deviceType. Device deviceType
  Device :: forall deviceType. deviceType -> Device deviceType
  deriving Show

class KnownDevice (device :: Device (DeviceType Nat)) where
  deviceVal :: Device (DeviceType Int16)

instance KnownDevice 'AnyDevice where
  deviceVal = AnyDevice

instance KnownDevice ('Device 'CPU) where
  deviceVal = Device CPU

instance (KnownNat deviceId) => KnownDevice ('Device ('CUDA deviceId)) where
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
