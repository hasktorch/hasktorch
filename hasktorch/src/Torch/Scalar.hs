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

instance Castable (ForeignPtr ATen.Scalar) Float where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable (ForeignPtr ATen.Scalar) Double where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable (ForeignPtr ATen.Scalar) Int where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

class (Castable (ForeignPtr ATen.Scalar) a) => Scalar a
instance Scalar Float
instance Scalar Double
instance Scalar Int

