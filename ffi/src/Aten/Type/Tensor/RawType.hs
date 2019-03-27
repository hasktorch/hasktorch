{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.Tensor.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawTensor

newtype Tensor = Tensor (Ptr RawTensor)
                   deriving (Eq, Ord, Show)

instance () => FPtr (Tensor) where
        type Raw Tensor = RawTensor
        get_fptr (Tensor ptr) = ptr
        cast_fptr_to_obj = Tensor
