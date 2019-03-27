{-# LANGUAGE EmptyDataDecls, ExistentialQuantification,
  FlexibleContexts, FlexibleInstances, ForeignFunctionInterface,
  MultiParamTypeClasses, ScopedTypeVariables, TypeFamilies,
  TypeSynonymInstances #-}
module Aten.Type.Storage.Interface where
import Data.Word
import Foreign.C
import Foreign.Ptr
import FFICXX.Runtime.Cast
import Aten.Type.Storage.RawType

class () => IStorage a where

upcastStorage :: forall a . (FPtr a, IStorage a) => a -> Storage
upcastStorage h
  = let fh = get_fptr h
        fh2 :: Ptr RawStorage = castPtr fh
      in cast_fptr_to_obj fh2

downcastStorage :: forall a . (FPtr a, IStorage a) => Storage -> a
downcastStorage h
  = let fh = get_fptr h
        fh2 = castPtr fh
      in cast_fptr_to_obj fh2
