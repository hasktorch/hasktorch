{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.Scalar.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.Scalar.RawType

class () => IScalar a where

upcastScalar :: forall a . (FPtr a, IScalar a) => a -> Scalar
upcastScalar h
  = let fh = get_fptr h
        fh2 :: Ptr RawScalar = castPtr fh
      in cast_fptr_to_obj fh2

downcastScalar :: forall a . (FPtr a, IScalar a) => Scalar -> a
downcastScalar h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
