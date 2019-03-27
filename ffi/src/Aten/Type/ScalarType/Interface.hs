{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.ScalarType.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.ScalarType.RawType

class () => IScalarType a where

upcastScalarType ::
                 forall a . (FPtr a, IScalarType a) => a -> ScalarType
upcastScalarType h
  = let fh = get_fptr h
        fh2 :: Ptr RawScalarType = castPtr fh
      in cast_fptr_to_obj fh2

downcastScalarType ::
                   forall a . (FPtr a, IScalarType a) => ScalarType -> a
downcastScalarType h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
