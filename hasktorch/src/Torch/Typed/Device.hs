{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Typed.Device where

import           GHC.TypeLits

import           Torch.Typed.Aux

data DeviceType = CPU | CUDA

data Device = Device { deviceType :: DeviceType, deviceIndex :: Int }

class KnownDevice (device :: (DeviceType, Nat)) where
  deviceVal :: Device

instance (KnownNat n) => KnownDevice '( 'CPU, n) where
  deviceVal = Device CPU (natValI @n)

instance (KnownNat n) => KnownDevice '( 'CUDA, n) where
  deviceVal = Device CUDA (natValI @n)
