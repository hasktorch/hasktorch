{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.SparseTensorRef.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawSparseTensorRef

newtype SparseTensorRef = SparseTensorRef (Ptr RawSparseTensorRef)
                            deriving (Eq, Ord, Show)

instance () => FPtr (SparseTensorRef) where
        type Raw SparseTensorRef = RawSparseTensorRef
        get_fptr (SparseTensorRef ptr) = ptr
        cast_fptr_to_obj = SparseTensorRef
