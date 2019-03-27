{-# LANGUAGE ForeignFunctionInterface, TypeFamilies,
  MultiParamTypeClasses, FlexibleInstances, TypeSynonymInstances,
  EmptyDataDecls, ExistentialQuantification, ScopedTypeVariables #-}
module Aten.Type.Storage.RawType where
import Foreign.Ptr
import FFICXX.Runtime.Cast

data RawStorage

newtype Storage = Storage (Ptr RawStorage)
                    deriving (Eq, Ord, Show)

instance () => FPtr (Storage) where
        type Raw Storage = RawStorage
        get_fptr (Storage ptr) = ptr
        cast_fptr_to_obj = Storage
