{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.IntArrayRef.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.IntArrayRef.RawType

class () => IIntArrayRef a where

upcastIntArrayRef ::
                  forall a . (FPtr a, IIntArrayRef a) => a -> IntArrayRef
upcastIntArrayRef h
  = let fh = get_fptr h
        fh2 :: Ptr RawIntArrayRef = castPtr fh
      in cast_fptr_to_obj fh2

downcastIntArrayRef ::
                    forall a . (FPtr a, IIntArrayRef a) => IntArrayRef -> a
downcastIntArrayRef h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
