{-# LANGUAGE FlexibleInstances, FlexibleContexts, TypeFamilies,
  MultiParamTypeClasses, OverlappingInstances, IncoherentInstances
  #-}
module Aten.Type.SparseTensorRef.Cast where
import Foreign.Ptr
import FFICXX.Runtime.Cast
import System.IO.Unsafe
import Aten.Type.SparseTensorRef.RawType
import Aten.Type.SparseTensorRef.Interface

instance (ISparseTensorRef a, FPtr a) =>
         Castable (a) (Ptr RawSparseTensorRef)
         where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))

instance () => Castable (SparseTensorRef) (Ptr RawSparseTensorRef)
         where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))
