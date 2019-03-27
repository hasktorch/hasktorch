{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.TensorOptions.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.TensorOptions.RawType
import Aten.Type.TensorOptions.FFI
import Aten.Type.TensorOptions.Interface
import Aten.Type.TensorOptions.Cast

instance () => ITensorOptions (TensorOptions) where

newTensorOptions :: () => IO TensorOptions
newTensorOptions = xformnull c_tensoroptions_newtensoroptions
