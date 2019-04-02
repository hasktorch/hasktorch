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

newTensorOptions :: () => CShort -> IO TensorOptions
newTensorOptions = xform0 c_tensoroptions_newtensoroptions

tensorOptions_dtype ::
                      () => TensorOptions -> CChar -> IO TensorOptions
tensorOptions_dtype = xform1 c_tensoroptions_tensoroptions_dtype
