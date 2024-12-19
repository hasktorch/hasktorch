{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Device where

import qualified Data.Int as I
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Type as ATen

data DeviceType = CPU | CUDA | MPS
  deriving (Eq, Ord, Show)

data Device = Device {deviceType :: DeviceType, deviceIndex :: I.Int16}
  deriving (Eq, Ord, Show)

instance Castable DeviceType ATen.DeviceType where
  cast CPU f = f ATen.kCPU
  cast CUDA f = f ATen.kCUDA
  cast MPS f = f ATen.kMPS

  uncast x f
    | x == ATen.kCPU = f CPU
    | x == ATen.kCUDA = f CUDA
    | x == ATen.kMPS = f MPS
