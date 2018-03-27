{-# LANGUAGE InstanceSigs #-}
module Torch.Indef.Storage.Copy where

import Foreign
import Foreign.C.Types
import Torch.Indef.Types

import Torch.Types.TH
import Torch.Sig.Types
import Control.Monad.IO.Class
import Control.Monad.Reader.Class
import qualified Foreign.Marshal.Array    as FM
import qualified Torch.Sig.Types          as Sig
import qualified Torch.Sig.State          as Sig
import qualified Torch.Sig.Storage        as Sig
import qualified Torch.Sig.Storage.Memory as Sig
import qualified Torch.Sig.Storage.Copy   as Sig
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

copyType
  :: (Ptr C'THState -> IO (Ptr a))
  -> (Ptr CState -> Ptr CStorage -> Ptr a -> IO ())
  -> Storage
  -> Torch (Ptr a)
copyType newPtr cfun t = do
  s <- ask
  liftIO $ do
    tar <- newPtr nullPtr
    withForeignPtr (Sig.asForeign s) $ \s' ->
      withForeignPtr (Sig.storage t) $ \t' ->
        cfun s' t' tar >> pure tar

instance Class.StorageCopyRaw Torch Storage where
  rawCopy :: Storage -> Torch [HsReal]
  rawCopy t = ask >>= \s -> liftIO $ do
    withForeignPtr (Sig.asForeign s) $ \s' ->
      withForeignPtr (Sig.storage t) $ \t' -> do
        sz  <- fromIntegral <$> Sig.c_size s' t'
        res <- FM.mallocArray (fromIntegral sz)
        Sig.c_rawCopy s' t' res
        (fmap.fmap) c2hsReal (FM.peekArray (fromIntegral sz) res)

  copy :: Storage -> Torch Storage
  copy t = do
    s <- ask
    fmap Sig.asStorage . liftIO $
      withForeignPtr (Sig.asForeign s) $ \s' -> do
        store <- Sig.c_new s'
        withForeignPtr (storage t) (\t' -> Sig.c_copy s' t' store)
        newForeignPtrEnv Sig.p_free s' store

  copyLong :: Storage -> Torch (Ptr CTHLongStorage)
  copyLong = copyType L.c_new Sig.c_copyLong

  copyFloat :: Storage -> Torch (Ptr CTHFloatStorage)
  copyFloat = copyType F.c_new Sig.c_copyFloat

  copyByte :: Storage -> Torch (Ptr CTHByteStorage)
  copyByte = copyType B.c_new Sig.c_copyByte

  -- copyChar :: Storage -> Torch (Ptr CTHCharStorage)
  -- copyChar = copyType C.c_new Sig.c_copyChar

  copyShort :: Storage -> Torch (Ptr CTHShortStorage)
  copyShort = copyType S.c_new Sig.c_copyShort

  copyInt :: Storage -> Torch (Ptr CTHIntStorage)
  copyInt = copyType I.c_new Sig.c_copyInt

  copyDouble :: Storage -> Torch (Ptr CTHDoubleStorage)
  copyDouble = copyType D.c_new Sig.c_copyDouble

  -- copyHalf   :: Storage -> Torch (Ptr CTHHalfStorage)
  -- copyHalf = copyType H.c_new Sig.c_copyHalf


