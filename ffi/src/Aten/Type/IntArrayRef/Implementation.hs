{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.IntArrayRef.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.IntArrayRef.RawType
import Aten.Type.IntArrayRef.FFI
import Aten.Type.IntArrayRef.Interface
import Aten.Type.IntArrayRef.Cast

instance () => IIntArrayRef (IntArrayRef) where

newIntArrayRef :: () => IO IntArrayRef
newIntArrayRef = xformnull c_intarrayref_newintarrayref
