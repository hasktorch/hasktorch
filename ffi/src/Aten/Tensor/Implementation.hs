{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Tensor.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Tensor.RawType
import Aten.Tensor.FFI
import Aten.Tensor.Interface
import Aten.Tensor.Cast

instance () => ITensor (Tensor) where

newTensor :: () => IO Tensor
newTensor = xformnull c_tensor_newtensor
