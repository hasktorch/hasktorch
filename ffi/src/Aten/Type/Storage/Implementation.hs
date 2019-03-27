{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.Storage.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.Storage.RawType
import Aten.Type.Storage.FFI
import Aten.Type.Storage.Interface
import Aten.Type.Storage.Cast

instance () => IStorage (Storage) where

newStorage :: () => IO Storage
newStorage = xformnull c_storage_newstorage
