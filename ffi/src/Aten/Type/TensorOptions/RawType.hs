{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.TensorOptions.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawTensorOptions

newtype TensorOptions = TensorOptions (Ptr RawTensorOptions)
                          deriving (Eq, Ord, Show)

instance () => FPtr (TensorOptions) where
        type Raw TensorOptions = RawTensorOptions
        get_fptr (TensorOptions ptr) = ptr
        cast_fptr_to_obj = TensorOptions
