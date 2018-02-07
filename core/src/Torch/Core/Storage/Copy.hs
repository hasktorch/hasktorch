{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Storage.Copy where

import Torch.Class.Internal (HsReal)
import Foreign
import Foreign.C.Types
import THTypes
import qualified StorageCopy as Sig
import qualified Storage as Sig
import qualified Torch.Class.Storage.Copy as Class

import qualified THLongStorage   as L
import qualified THFloatStorage  as F
import qualified THByteStorage   as B
-- import qualified THCharStorage   as C
import qualified THShortStorage  as S
import qualified THIntStorage    as I
import qualified THDoubleStorage as D
-- import qualified THHalfStorage   as H

import Torch.Core.Types

copyType :: IO (Ptr a) -> (Ptr CStorage -> Ptr a -> IO ()) -> Storage -> IO (Ptr a)
copyType newPtr cfun t = do
  tar <- newPtr
  withForeignPtr (storage t) (`cfun` tar)
  pure tar

instance Class.StorageCopy Storage where
  -- rawCopy :: Storage -> IO (Ptr (HsReal Storage))
  copy :: Storage -> IO Storage
  copy t = do
    tar <- Sig.c_new
    withForeignPtr (storage t) (`Sig.c_copy` tar)
    Storage <$> newForeignPtr Sig.p_free tar

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
  -- copyHalf = copyType F.c_new Sig.c_copyHalf


