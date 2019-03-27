{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.TensorList.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.TensorList.RawType
import Aten.Type.TensorList.FFI
import Aten.Type.TensorList.Interface
import Aten.Type.TensorList.Cast

instance () => ITensorList (TensorList) where

newTensorList :: () => IO TensorList
newTensorList = xformnull c_tensorlist_newtensorlist
