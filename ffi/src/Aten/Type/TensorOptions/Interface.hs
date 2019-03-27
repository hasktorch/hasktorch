{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.TensorOptions.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.TensorOptions.RawType

class () => ITensorOptions a where

upcastTensorOptions ::
                    forall a . (FPtr a, ITensorOptions a) => a -> TensorOptions
upcastTensorOptions h
  = let fh = get_fptr h
        fh2 :: Ptr RawTensorOptions = castPtr fh
      in cast_fptr_to_obj fh2

downcastTensorOptions ::
                      forall a . (FPtr a, ITensorOptions a) => TensorOptions -> a
downcastTensorOptions h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
