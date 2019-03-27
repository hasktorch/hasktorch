{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.ScalarType.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawScalarType

newtype ScalarType = ScalarType (Ptr RawScalarType)
                       deriving (Eq, Ord, Show)

instance () => FPtr (ScalarType) where
        type Raw ScalarType = RawScalarType
        get_fptr (ScalarType ptr) = ptr
        cast_fptr_to_obj = ScalarType
