{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Scalar where

import Foreign.ForeignPtr

import qualified ATen.Const as ATen
import qualified ATen.Managed.Type.Scalar as ATen
import qualified ATen.Type as ATen
import ATen.Managed.Cast
import ATen.Class (Castable(..))
import ATen.Cast

instance Castable Double (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable Int (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

class (Castable a (ForeignPtr ATen.Scalar)) => Scalar a
instance Scalar Double
instance Scalar Int

