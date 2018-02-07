{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Copy where

import Foreign
import Foreign.C.Types
import THTypes
import qualified Tensor     as Sig
import qualified TensorCopy as Sig
import qualified Torch.Class.Tensor.Copy as Class

import qualified THLongTensor   as L
import qualified THFloatTensor  as F
import qualified THByteTensor   as B
-- import qualified THCharTensor   as C
import qualified THShortTensor  as S
import qualified THIntTensor    as I
import qualified THDoubleTensor as D
-- import qualified THHalfTensor   as H

import Torch.Core.Types

copyType :: IO (Ptr a) -> (Ptr CTensor -> Ptr a -> IO ()) -> Tensor -> IO (Ptr a)
copyType newPtr cfun t = do
  tar <- newPtr
  withForeignPtr (tensor t) (`cfun` tar)
  pure tar

instance Class.TensorCopy Tensor where
  copy :: Tensor -> IO Tensor
  copy t = do
    tar <- Sig.c_new
    withForeignPtr (tensor t) (`Sig.c_copy` tar)
    Tensor <$> newForeignPtr Sig.p_free tar

  copyLong :: Tensor -> IO (Ptr CTHLongTensor)
  copyLong = copyType L.c_new Sig.c_copyLong

  copyFloat :: Tensor -> IO (Ptr CTHFloatTensor)
  copyFloat = copyType F.c_new Sig.c_copyFloat

  copyByte :: Tensor -> IO (Ptr CTHByteTensor)
  copyByte = copyType B.c_new Sig.c_copyByte

  -- copyChar :: Tensor -> IO (Ptr CTHCharTensor)
  -- copyChar = copyType C.c_new Sig.c_copyChar

  copyShort :: Tensor -> IO (Ptr CTHShortTensor)
  copyShort = copyType S.c_new Sig.c_copyShort

  copyInt :: Tensor -> IO (Ptr CTHIntTensor)
  copyInt = copyType I.c_new Sig.c_copyInt

  copyDouble :: Tensor -> IO (Ptr CTHDoubleTensor)
  copyDouble = copyType D.c_new Sig.c_copyDouble

  -- copyHalf   :: Tensor -> IO (Ptr CTHHalfTensor)
  -- copyHalf = copyType H.c_new Sig.c_copyHalf

