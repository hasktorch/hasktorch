{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.SparseTensorRef.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.SparseTensorRef.RawType
import {-# SOURCE #-} Aten.Type.Tensor.Interface

class () => ISparseTensorRef a where

upcastSparseTensorRef ::
                      forall a . (FPtr a, ISparseTensorRef a) => a -> SparseTensorRef
upcastSparseTensorRef h
  = let fh = get_fptr h
        fh2 :: Ptr RawSparseTensorRef = castPtr fh
      in cast_fptr_to_obj fh2

downcastSparseTensorRef ::
                        forall a . (FPtr a, ISparseTensorRef a) => SparseTensorRef -> a
downcastSparseTensorRef h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
