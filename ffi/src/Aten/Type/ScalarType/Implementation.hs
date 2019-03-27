{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.ScalarType.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.ScalarType.RawType
import Aten.Type.ScalarType.FFI
import Aten.Type.ScalarType.Interface
import Aten.Type.ScalarType.Cast

instance () => IScalarType (ScalarType) where

newScalarType :: () => IO ScalarType
newScalarType = xformnull c_scalartype_newscalartype
