{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.Tensor.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.Tensor.RawType

class () => ITensor a where

upcastTensor :: forall a . (FPtr a, ITensor a) => a -> Tensor
upcastTensor h
  = let fh = get_fptr h
        fh2 :: Ptr RawTensor = castPtr fh
      in cast_fptr_to_obj fh2

downcastTensor :: forall a . (FPtr a, ITensor a) => Tensor -> a
downcastTensor h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
