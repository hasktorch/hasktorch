{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Scalar where

import Foreign.ForeignPtr
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import Torch.Internal.Managed.Cast
import qualified Torch.Internal.Managed.Type.Scalar as ATen
import qualified Torch.Internal.Type as ATen

instance Castable Float (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_f (realToFrac x) >>= f
  uncast x f = undefined

instance Castable Double (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_d (realToFrac x) >>= f
  uncast x f = undefined

instance Castable Int (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_i (fromIntegral x) >>= f
  uncast x f = undefined

instance Castable Bool (ForeignPtr ATen.Scalar) where
  cast x f = ATen.newScalar_b (if x then 1 else 0) >>= f
  uncast x f = undefined

class (Castable a (ForeignPtr ATen.Scalar)) => Scalar a

instance Scalar Float

instance Scalar Double

instance Scalar Int

instance Scalar Bool
