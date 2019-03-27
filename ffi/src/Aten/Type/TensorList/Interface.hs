{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.TensorList.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.TensorList.RawType

class () => ITensorList a where

upcastTensorList ::
                 forall a . (FPtr a, ITensorList a) => a -> TensorList
upcastTensorList h
  = let fh = get_fptr h
        fh2 :: Ptr RawTensorList = castPtr fh
      in cast_fptr_to_obj fh2

downcastTensorList ::
                   forall a . (FPtr a, ITensorList a) => TensorList -> a
downcastTensorList h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
