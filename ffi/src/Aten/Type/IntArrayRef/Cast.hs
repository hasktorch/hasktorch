{-# LANGUAGE FlexibleInstances, FlexibleContexts, TypeFamilies,
  MultiParamTypeClasses, OverlappingInstances, IncoherentInstances
  #-}
module Aten.Type.IntArrayRef.Cast where
import Foreign.Ptr
import FFICXX.Runtime.Cast
import System.IO.Unsafe
import Aten.Type.IntArrayRef.RawType
import Aten.Type.IntArrayRef.Interface

instance (IIntArrayRef a, FPtr a) =>
         Castable (a) (Ptr RawIntArrayRef)
         where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))

instance () => Castable (IntArrayRef) (Ptr RawIntArrayRef) where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))
