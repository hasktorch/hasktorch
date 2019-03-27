{-# LANGUAGE FlexibleInstances, FlexibleContexts, TypeFamilies,
  MultiParamTypeClasses, OverlappingInstances, IncoherentInstances
  #-}
module Aten.Type.ScalarType.Cast where
import Foreign.Ptr
import FFICXX.Runtime.Cast
import System.IO.Unsafe
import Aten.Type.ScalarType.RawType
import Aten.Type.ScalarType.Interface

instance (IScalarType a, FPtr a) =>
         Castable (a) (Ptr RawScalarType)
         where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))

instance () => Castable (ScalarType) (Ptr RawScalarType) where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))
