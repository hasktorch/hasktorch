{-# LANGUAGE FlexibleInstances, FlexibleContexts, TypeFamilies,
  MultiParamTypeClasses, OverlappingInstances, IncoherentInstances
  #-}
module Aten.Type.TensorOptions.Cast where
import Foreign.Ptr
import FFICXX.Runtime.Cast
import System.IO.Unsafe
import Aten.Type.TensorOptions.RawType
import Aten.Type.TensorOptions.Interface

instance (ITensorOptions a, FPtr a) =>
         Castable (a) (Ptr RawTensorOptions)
         where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))

instance () => Castable (TensorOptions) (Ptr RawTensorOptions)
         where
        cast x f = f (castPtr (get_fptr x))
        uncast x f = f (cast_fptr_to_obj (castPtr x))
