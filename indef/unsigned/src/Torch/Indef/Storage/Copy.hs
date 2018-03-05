{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Storage.Copy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import qualified Foreign.Marshal.Array as FM
import qualified Torch.Signature.Types        as Sig
import qualified Torch.Signature.Storage      as Sig
import qualified Torch.Signature.Storage.Copy as Sig
import qualified Torch.Class.Storage.Copy as Class

import qualified Torch.FFI.TH.Long.Storage   as L
import qualified Torch.FFI.TH.Float.Storage  as F
import qualified Torch.FFI.TH.Byte.Storage   as B
-- import qualified THCharStorage   as C
import qualified Torch.FFI.TH.Short.Storage  as S
import qualified Torch.FFI.TH.Int.Storage    as I
import qualified Torch.FFI.TH.Double.Storage as D
-- import qualified THHalfStorage   as H

import Torch.Indef.Types

copyType :: IO (Ptr a) -> (Ptr CStorage -> Ptr a -> IO ()) -> Storage -> IO (Ptr a)
copyType newPtr cfun t = do
  tar <- newPtr
  withForeignPtr (storage t) (`cfun` tar)
  pure tar

instance Class.StorageCopy Storage where
  rawCopy :: Storage -> IO [HsReal]
  rawCopy s = do
    sz <- withForeignPtr (storage s) (\s' -> fromIntegral <$> Sig.c_size s')
    res <- FM.mallocArray (fromIntegral sz)
    withForeignPtr (storage s) (`Sig.c_rawCopy` res)
    (fmap.fmap) c2hsReal (FM.peekArray (fromIntegral sz) res)

  copy :: Storage -> IO Storage
  copy t = do
    tar <- Sig.c_new
    withForeignPtr (storage t) (`Sig.c_copy` tar)
    Sig.asStorage <$> newForeignPtr Sig.p_free tar

  copyLong :: Storage -> IO (Ptr CTHLongStorage)
  copyLong = copyType L.c_new Sig.c_copyLong

  copyFloat :: Storage -> IO (Ptr CTHFloatStorage)
  copyFloat = copyType F.c_new Sig.c_copyFloat

  copyByte :: Storage -> IO (Ptr CTHByteStorage)
  copyByte = copyType B.c_new Sig.c_copyByte

  -- copyChar :: Storage -> IO (Ptr CTHCharStorage)
  -- copyChar = copyType C.c_new Sig.c_copyChar

  copyShort :: Storage -> IO (Ptr CTHShortStorage)
  copyShort = copyType S.c_new Sig.c_copyShort

  copyInt :: Storage -> IO (Ptr CTHIntStorage)
  copyInt = copyType I.c_new Sig.c_copyInt

  copyDouble :: Storage -> IO (Ptr CTHDoubleStorage)
  copyDouble = copyType D.c_new Sig.c_copyDouble

  -- copyHalf   :: Storage -> IO (Ptr CTHHalfStorage)
  -- copyHalf = copyType H.c_new Sig.c_copyHalf


