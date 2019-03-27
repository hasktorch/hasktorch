{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.SparseTensorRef.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.SparseTensorRef.RawType
import Aten.Type.SparseTensorRef.FFI
import Aten.Type.SparseTensorRef.Interface
import Aten.Type.SparseTensorRef.Cast
import Aten.Type.Tensor.RawType
import Aten.Type.Tensor.Cast
import Aten.Type.Tensor.Interface

instance () => ISparseTensorRef (SparseTensorRef) where

newSparseTensorRef ::
                     (ITensor c0, FPtr c0) => c0 -> IO SparseTensorRef
newSparseTensorRef = xform0 c_sparsetensorref_newsparsetensorref
