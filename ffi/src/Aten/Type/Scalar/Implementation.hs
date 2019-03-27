{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.Scalar.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.Scalar.RawType
import Aten.Type.Scalar.FFI
import Aten.Type.Scalar.Interface
import Aten.Type.Scalar.Cast

instance () => IScalar (Scalar) where

newScalar :: () => IO Scalar
newScalar = xformnull c_scalar_newscalar
