{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Device where

import qualified Data.Int as I

import           LibTorch.ATen.Class                     ( Castable(..) )
import qualified LibTorch.ATen.Const                    as ATen
import qualified LibTorch.ATen.Type                     as ATen

data DeviceType = CPU | CUDA
  deriving (Eq, Ord, Show)

data Device = Device { deviceType :: DeviceType, deviceIndex :: I.Int16 }
  deriving (Eq, Ord, Show)

instance Castable DeviceType ATen.DeviceType where
  cast CPU   f = f ATen.kCPU
  cast CUDA  f = f ATen.kCUDA

  uncast x f
    | x == ATen.kCPU = f CPU
    | x == ATen.kCUDA = f CUDA
