{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.TensorList.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawTensorList

newtype TensorList = TensorList (Ptr RawTensorList)
                       deriving (Eq, Ord, Show)

instance () => FPtr (TensorList) where
        type Raw TensorList = RawTensorList
        get_fptr (TensorList ptr) = ptr
        cast_fptr_to_obj = TensorList
