-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Copy
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Functions to copy (and cast) tensors into different types.
-- This is a pure module.
-------------------------------------------------------------------------------
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Dynamic.Tensor.Copy
  ( copy
  , copyByte
  , copyChar
  , copyShort
  , copyInt
  , copyLong
  , copyFloat
  , copyDouble
  ) where

import Foreign
import Foreign.C.Types
import Data.List (intercalate)
import Control.Exception.Safe (throwString)
import System.IO.Unsafe
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Tensor as Sig
import qualified Torch.Sig.Tensor.Copy as Sig

import qualified Torch.FFI.TH.Byte.Tensor   as B
import qualified Torch.FFI.TH.Short.Tensor  as S
import qualified Torch.FFI.TH.Int.Tensor    as I
import qualified Torch.FFI.TH.Long.Tensor   as L
import qualified Torch.FFI.TH.Char.Tensor   as C
import qualified Torch.FFI.TH.Float.Tensor  as F
import qualified Torch.FFI.TH.Double.Tensor as D

import Torch.Indef.Types

copyType
  :: IO (Ptr a)
  -> FinalizerPtr a
  -> (ForeignPtr C'THState -> ForeignPtr a -> b)
  -> (Ptr CState -> Ptr CTensor -> Ptr a -> IO ())
  -> (Ptr C'THState -> Ptr a -> Ptr TH.C'THLongStorage -> Ptr TH.C'THLongStorage -> IO ())
  -> Dynamic -> b
copyType newPtr fin builder cfun resizer t = unsafePerformIO . withDynamicState t $ \s' t' -> do
  target <- newPtr

  resizer
    <$> TH.newCState
    <*> pure target
    <*> Sig.c_newSizeOf s' t'
    <*> Sig.c_newStrideOf s' t'

  builder
    <$> (TH.newCState >>= TH.manageState)
    <*> newForeignPtr fin target
{-# NOINLINE copyType #-}

-- | Copy a tensor.
copy :: Dynamic -> Dynamic
copy t = unsafePerformIO . withDynamicState t $ \s' t' -> do
  target <- Sig.c_new s'
  Sig.c_resizeAs s' target t'
  Sig.c_copy s' target t'
  mkDynamic s' target
{-# NOINLINE copy #-}

-- | copy a tensor to a byte tensor. *Use at your own discresion*
copyByte   = copyType B.c_new_ B.p_free   TH.byteDynamic Sig.c_copyByte B.c_resize
-- | copy a tensor to a char tensor. *Use at your own discresion*
copyChar   = copyType C.c_new_ C.p_free   TH.charDynamic Sig.c_copyChar C.c_resize
-- | copy a tensor to a short tensor. *Use at your own discresion*
copyShort  = copyType S.c_new_ S.p_free  TH.shortDynamic Sig.c_copyShort S.c_resize
-- | copy a tensor to a int tensor. *Use at your own discresion*
copyInt    = copyType I.c_new_ I.p_free    TH.intDynamic Sig.c_copyInt I.c_resize
-- | copy a tensor to a long tensor. *Use at your own discresion*
copyLong   = copyType L.c_new_ L.p_free   TH.longDynamic Sig.c_copyLong L.c_resize
-- | copy a tensor to a float tensor. *Use at your own discresion*
copyFloat  = copyType F.c_new_ F.p_free  TH.floatDynamic Sig.c_copyFloat F.c_resize
-- | copy a tensor to a double tensor. *Use at your own discresion*
copyDouble = copyType D.c_new_ D.p_free TH.doubleDynamic Sig.c_copyDouble D.c_resize

-- FIXME: reintroduce Half
-- copyHalf   :: t -> io H.Dynamic

-- #if CUDA
-- class GPUTensorCopy gpu cpu | gpu -> cpu where
--   copyCuda             :: gpu -> io gpu
--   copyIgnoringOverlaps :: gpu -> io gpu
-- 
--   copyCudaByte    :: gpu -> IO Cuda.ByteDynamic
--   copyCudaChar    :: gpu -> IO Cuda.CharDynamic
--   copyCudaShort   :: gpu -> IO Cuda.ShortDynamic
--   copyCudaInt     :: gpu -> IO Cuda.IntDynamic
--   copyCudaLong    :: gpu -> IO Cuda.LongDynamic
--   copyCudaDouble  :: gpu -> IO Cuda.DoubleDynamic
-- 
--   copyCPU         :: gpu -> IO cpu
--   copyAsyncCPU    :: gpu -> IO cpu
-- 
--   thCopyCuda      :: cpu -> IO gpu
--   thCopyAsyncCuda :: cpu -> IO gpu
-- #endif
