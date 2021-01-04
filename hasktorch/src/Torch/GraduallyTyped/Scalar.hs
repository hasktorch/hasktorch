{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.GraduallyTyped.Scalar where

import Foreign.ForeignPtr (ForeignPtr)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Type.Scalar as ATen
import qualified Torch.Internal.Type as ATen

instance Castable Float (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast _x _f = undefined

instance Castable Double (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast _x _f = undefined

instance Castable Int (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast _x _f = undefined

instance Castable Integer (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast _x _f = undefined

class (Castable a (ForeignPtr ATen.Scalar)) => Scalar a

instance Scalar Float

instance Scalar Double

instance Scalar Int

instance Scalar Integer
