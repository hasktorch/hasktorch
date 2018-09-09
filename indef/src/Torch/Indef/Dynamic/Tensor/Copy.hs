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
{-# OPTIONS_GHC -fno-cse #-}
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
import Torch.Types.TH (C'THState)
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Tensor as Sig
import qualified Torch.Sig.Types.Global as Sig
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
  :: (Ptr TH.C'THLongStorage -> Ptr TH.C'THLongStorage -> IO (Ptr a))
  -> FinalizerPtr a
  -> (ForeignPtr C'THState -> ForeignPtr a -> b)

  -> (Ptr CState -> Ptr CTensor -> Ptr a -> IO ())
  -> Dynamic -> b
copyType newWithSize_ fin builder cfun t = unsafePerformIO . withDynamicState t $ \s' t' -> do
    sizes   <- Sig.c_newSizeOf s' t'
    strides <- Sig.c_newStrideOf s' t'
    target  <- newWithSize_ sizes strides

    cfun s' t' target

    builder TH.torchstate <$> newForeignPtr fin target
{-# NOINLINE copyType #-}

-- | Copy a tensor.
copy :: Dynamic -> Dynamic
copy t = unsafePerformIO . withDynamicState t $ \s' t' -> do
  target <- Sig.c_new s'
  Sig.c_resizeAs s' target t'
  Sig.c_copy s' target t'
  mkDynamic target
{-# NOINLINE copy #-}

-- | copy a tensor to a byte tensor. *Use at your own discresion*
copyByte :: Dynamic -> TH.ByteDynamic
copyByte = copyType B.c_newWithSize_ B.p_free TH.byteDynamic Sig.c_copyByte
-- | copy a tensor to a char tensor. *Use at your own discresion*
copyChar :: Dynamic -> TH.CharDynamic
copyChar   = copyType C.c_newWithSize_ C.p_free TH.charDynamic Sig.c_copyChar
-- | copy a tensor to a short tensor. *Use at your own discresion*
copyShort :: Dynamic -> TH.ShortDynamic
copyShort  = copyType S.c_newWithSize_ S.p_free TH.shortDynamic Sig.c_copyShort
-- | copy a tensor to a int tensor. *Use at your own discresion*
copyInt :: Dynamic -> TH.IntDynamic
copyInt    = copyType I.c_newWithSize_ I.p_free TH.intDynamic Sig.c_copyInt
-- | copy a tensor to a long tensor. *Use at your own discresion*
copyLong :: Dynamic -> TH.LongDynamic
copyLong   = copyType L.c_newWithSize_ L.p_free TH.longDynamic Sig.c_copyLong
-- | copy a tensor to a float tensor. *Use at your own discresion*
copyFloat :: Dynamic -> TH.FloatDynamic
copyFloat  = copyType F.c_newWithSize_ F.p_free TH.floatDynamic Sig.c_copyFloat
-- | copy a tensor to a double tensor. *Use at your own discresion*
copyDouble :: Dynamic -> TH.DoubleDynamic
copyDouble  = copyType D.c_newWithSize_ D.p_free TH.doubleDynamic Sig.c_copyDouble
-- copyDouble :: Dynamic -> TH.DoubleDynamic
-- copyDouble = copyType D.c_new_ D.p_free TH.doubleDynamic Sig.c_copyDouble D.c_resize

-- copyDouble :: Dynamic -> TH.DoubleDynamic
-- copyDouble t = unsafePerformIO . withDynamicState t $ \s' t' -> do
--   withForeignPtr TH.torchstate $ \ths' -> do
--     sizes   <- Sig.c_newSizeOf s' t'
--     strides <- Sig.c_newStrideOf s' t'
--     target  <- D.c_newWithSize ths' sizes strides
--
--     -- mapM (size t . fromIntegral) [0.. nDimension t - 1] >>=
--     Sig.c_copyDouble s' t' target
--
--     out <- TH.doubleDynamic TH.torchstate <$> newForeignPtr D.p_free target
--     pure out


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
