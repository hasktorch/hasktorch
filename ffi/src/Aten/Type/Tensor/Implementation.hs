{-# LANGUAGE EmptyDataDecls, FlexibleContexts, FlexibleInstances,
  ForeignFunctionInterface, IncoherentInstances,
  MultiParamTypeClasses, OverlappingInstances, TemplateHaskell,
  TypeFamilies, TypeSynonymInstances #-}
module Aten.Type.Tensor.Implementation where
import Data.Monoid
import Data.Word
import Foreign.C
import Foreign.Ptr
import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import System.IO.Unsafe
import FFICXX.Runtime.Cast
import FFICXX.Runtime.TH
import Aten.Type.Tensor.RawType
import Aten.Type.Tensor.FFI
import Aten.Type.Tensor.Interface
import Aten.Type.Tensor.Cast

instance () => ITensor (Tensor) where

newTensor :: () => IO Tensor
newTensor = xformnull c_tensor_newtensor

tensor_dim :: () => Tensor -> IO CLong
tensor_dim = xform0 c_tensor_tensor_dim

tensor_storage_offset :: () => Tensor -> IO CLong
tensor_storage_offset = xform0 c_tensor_tensor_storage_offset

tensor_defined :: () => Tensor -> IO CInt
tensor_defined = xform0 c_tensor_tensor_defined

tensor_reset :: () => Tensor -> IO ()
tensor_reset = xform0 c_tensor_tensor_reset

tensor_cpu :: () => Tensor -> IO ()
tensor_cpu = xform0 c_tensor_tensor_cpu

tensor_cuda :: () => Tensor -> IO ()
tensor_cuda = xform0 c_tensor_tensor_cuda

tensor_print :: () => Tensor -> IO ()
tensor_print = xform0 c_tensor_tensor_print
