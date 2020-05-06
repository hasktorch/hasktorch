{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Scalar where

import Foreign.ForeignPtr
import Data.Int
import Data.Word

import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Type.Scalar as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Internal.Managed.Cast
import Torch.Internal.Class (Castable(..))
import Torch.Internal.Cast

instance Castable Float (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable Double (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable Int (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

instance Castable Int64 (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

instance Castable Int32 (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

instance Castable Int16 (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

instance Castable Int8 (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

instance Castable Word8 (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

instance Castable Bool (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

class (Castable a (ForeignPtr ATen.Scalar)) => Scalar a
instance Scalar Float
instance Scalar Double
instance Scalar Int
instance Scalar Int32
instance Scalar Int16
instance Scalar Int8
instance Scalar Word8
instance Scalar Bool
