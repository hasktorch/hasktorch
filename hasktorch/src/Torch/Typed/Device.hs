{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.Typed.Device where

import           GHC.TypeLits

data DeviceType = CPU | CUDA

data Device = Device { deviceType :: DeviceType, deviceIndex :: Int }

class KnownDevice (device :: (DeviceType, Nat)) where
  deviceVal :: Device
