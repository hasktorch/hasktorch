{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Scalar where

import Foreign.ForeignPtr

import qualified LibTorch.ATen.Const as ATen
import qualified LibTorch.ATen.Managed.Type.Scalar as ATen
import qualified LibTorch.ATen.Type as ATen
import LibTorch.ATen.Managed.Cast
import LibTorch.ATen.Class (Castable(..))
import LibTorch.ATen.Cast

instance Castable Float (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable Double (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable Int (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

class (Castable a (ForeignPtr ATen.Scalar)) => Scalar a
instance Scalar Float
instance Scalar Double
instance Scalar Int

