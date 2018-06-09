-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Storage.Copy
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Conversions between two Storage types. Behaviour is not yet specified as to
-- what happens when you copy between incompatible types.
-------------------------------------------------------------------------------
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Indef.Storage.Copy
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
import Foreign.Ptr
import qualified Torch.Types.TH           as TH
import qualified Foreign.Marshal.Array    as FM
import qualified Torch.Sig.Types          as Sig
import qualified Torch.Sig.Types.Global   as Sig
import qualified Torch.Sig.Storage        as Sig
import qualified Torch.Sig.Storage.Memory as Sig
import qualified Torch.Sig.Storage.Copy   as Sig

import qualified Torch.FFI.TH.Long.Storage   as L
import qualified Torch.FFI.TH.Float.Storage  as F
import qualified Torch.FFI.TH.Byte.Storage   as B
import qualified Torch.FFI.TH.Char.Storage   as C
import qualified Torch.FFI.TH.Short.Storage  as S
import qualified Torch.FFI.TH.Int.Storage    as I
import qualified Torch.FFI.TH.Double.Storage as D

import Torch.Indef.Types

copyType
  :: IO (Ptr a)
  -> FinalizerPtr a
  -> (ForeignPtr TH.C'THState -> ForeignPtr a -> b)
  -> (Ptr CState -> Ptr CStorage -> Ptr a -> IO ())
  -> Storage -> IO b
copyType newPtr fin builder cfun t = withStorageState t $ \s' t' -> do
  target <- newPtr
  -- throwString $ intercalate ""
  --   [ "'hasktorch-indef-unsigned:Torch.Indef.Tensor.Dynamic.Copy.copyType':"
  --   , "must resize the target tensor before continuing"
  --   ]
  -- Sig.c_resizeAs s' target t'       -- << THIS NEEDS TO BE REMAPPED TO TENSORLONG SIZES
  cfun s' t' target

  builder
    <$> (TH.newCState >>= TH.manageState)
    <*> newForeignPtr fin target


rawCopy :: Storage -> IO [HsReal]
rawCopy t = withStorageState t $ \s' t' -> do
  sz  <- fromIntegral <$> Sig.c_size s' t'
  res <- FM.mallocArray (fromIntegral sz)
  Sig.c_rawCopy s' t' res
  (fmap.fmap) c2hsReal (FM.peekArray (fromIntegral sz) res)

-- | Copy a Storage object.
copy :: Storage -> IO Storage
copy t = withStorageState t $ \s' t' -> do
  store <- Sig.c_new s'
  Sig.c_copy s' t' store
  mkStorage s' store

-- | Copy a 'Storage' object to a CPU-backed LongStorage.
copyLong   = copyType L.c_new_ L.p_free TH.longStorage   Sig.c_copyLong

-- | Copy a 'Storage' object to a CPU-backed FloatStorage.
copyFloat  = copyType F.c_new_ F.p_free TH.floatStorage  Sig.c_copyFloat

-- | Copy a 'Storage' object to a CPU-backed ByteStorage.
copyByte   = copyType B.c_new_ B.p_free TH.byteStorage   Sig.c_copyByte

-- | Copy a 'Storage' object to a CPU-backed CharStorage.
copyChar   = copyType C.c_new_ C.p_free TH.charStorage   Sig.c_copyChar

-- | Copy a 'Storage' object to a CPU-backed ShortStorage.
copyShort  = copyType S.c_new_ S.p_free TH.shortStorage  Sig.c_copyShort

-- | Copy a 'Storage' object to a CPU-backed IntStorage.
copyInt    = copyType I.c_new_ I.p_free TH.intStorage    Sig.c_copyInt

-- | Copy a 'Storage' object to a CPU-backed DoubleStorage.
copyDouble = copyType D.c_new_ D.p_free TH.doubleStorage Sig.c_copyDouble

-- FIXME: reintroduce half
-- copyHalf   :: t -> IO HalfStorage

-- class GPUStorageCopy gpu cpu | gpu -> cpu where
--   thCopyCuda :: cpu -> IO gpu
--   copyCuda   :: gpu -> IO gpu
--   copyCPU    :: gpu -> IO cpu

