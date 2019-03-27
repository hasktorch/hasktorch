{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.IntArrayRef.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawIntArrayRef

newtype IntArrayRef = IntArrayRef (Ptr RawIntArrayRef)
                        deriving (Eq, Ord, Show)

instance () => FPtr (IntArrayRef) where
        type Raw IntArrayRef = RawIntArrayRef
        get_fptr (IntArrayRef ptr) = ptr
        cast_fptr_to_obj = IntArrayRef
