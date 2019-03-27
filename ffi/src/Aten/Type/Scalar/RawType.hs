{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.Scalar.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawScalar

newtype Scalar = Scalar (Ptr RawScalar)
                   deriving (Eq, Ord, Show)

instance () => FPtr (Scalar) where
        type Raw Scalar = RawScalar
        get_fptr (Scalar ptr) = ptr
        cast_fptr_to_obj = Scalar
