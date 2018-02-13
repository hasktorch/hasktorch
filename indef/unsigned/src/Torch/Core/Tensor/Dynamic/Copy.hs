{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Copy where

import Foreign
import Foreign.C.Types
import THTypes
import qualified Tensor     as Sig
import qualified TensorCopy as Sig
import qualified Torch.Class.C.Tensor.Copy as Class

import qualified THByteTensor   as B
import qualified THByteTypes    as B
import qualified THShortTensor  as S
import qualified THShortTypes   as S
import qualified THIntTensor    as I
import qualified THIntTypes     as I
import qualified THLongTensor   as L
import qualified THLongTypes    as L
import qualified THFloatTensor  as F
import qualified THFloatTypes   as F
import qualified THDoubleTensor as D
import qualified THDoubleTypes  as D

import Torch.Core.Types

copyType :: IO (Ptr a) -> FinalizerPtr a -> (Ptr CTensor -> Ptr a -> IO ()) -> Tensor -> IO (ForeignPtr a)
copyType newPtr fin cfun t = do
  tar <- newPtr
  withForeignPtr (tensor t) (`cfun` tar)
  newForeignPtr fin tar

instance Class.TensorCopy Tensor where
  copy :: Tensor -> IO Tensor
  copy t = do
    tar <- Sig.c_new
    withForeignPtr (tensor t) (`Sig.c_copy` tar)
    asDyn <$> newForeignPtr Sig.p_free tar

  copyByte :: Tensor -> IO B.DynTensor
  copyByte t = B.asDyn <$> (copyType B.c_new B.p_free Sig.c_copyByte t)

  -- copyChar :: Tensor -> IO (Ptr CTHCharTensor)
  -- copyChar = copyType C.c_new Sig.c_copyChar

  copyShort :: Tensor -> IO S.DynTensor
  copyShort t = S.asDyn <$> (copyType S.c_new S.p_free Sig.c_copyShort t)

  copyInt :: Tensor -> IO I.DynTensor
  copyInt t = I.asDyn <$> (copyType I.c_new I.p_free Sig.c_copyInt t)

  copyLong :: Tensor -> IO L.DynTensor
  copyLong t = L.asDyn <$> (copyType L.c_new L.p_free Sig.c_copyLong t)

  -- copyHalf   :: Tensor -> IO (Ptr CTHHalfTensor)
  -- copyHalf = copyType H.c_new Sig.c_copyHalf

  copyFloat :: Tensor -> IO F.DynTensor
  copyFloat t = F.asDyn <$> (copyType F.c_new F.p_free Sig.c_copyFloat t)

  copyDouble :: Tensor -> IO D.DynTensor
  copyDouble t = D.asDyn <$> (copyType D.c_new D.p_free Sig.c_copyDouble t)


