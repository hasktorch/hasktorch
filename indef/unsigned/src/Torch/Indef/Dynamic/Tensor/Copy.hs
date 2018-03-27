{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Dynamic.Tensor.Copy where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Control.Exception.Safe (throwString)
import qualified Torch.Sig.IsTensor as Sig
import qualified Torch.Sig.Tensor.Copy as Sig
import qualified Torch.Class.Tensor.Copy as Class
import qualified Torch.Class.IsTensor as Class

import qualified Torch.FFI.TH.Byte.Tensor   as B
import qualified Torch.Types.TH.Byte    as B
import qualified Torch.FFI.TH.Short.Tensor  as S
import qualified Torch.Types.TH.Short   as S
import qualified Torch.FFI.TH.Int.Tensor    as I
import qualified Torch.Types.TH.Int     as I
import qualified Torch.FFI.TH.Long.Tensor   as L
import qualified Torch.Types.TH.Long    as L
import qualified Torch.FFI.TH.Float.Tensor  as F
import qualified Torch.Types.TH.Float   as F
import qualified Torch.FFI.TH.Double.Tensor as D
import qualified Torch.Types.TH.Double  as D

import Torch.Indef.Types

copyType :: IO (Ptr a) -> FinalizerPtr a -> (Ptr CTensor -> Ptr a -> IO ()) -> Tensor -> IO (ForeignPtr a)
copyType newPtr fin cfun t = do
  tar <- newPtr
  throwString "'hasktorch-indef-unsigned:Torch.Indef.Tensor.Dynamic.Copy.copyType': must resize the target tensor before continuing"
  withForeignPtr (tensor t) (`cfun` tar)
  newForeignPtr fin tar

instance Class.TensorCopy Tensor where
  copy :: Tensor -> IO Tensor
  copy t = do
    tar <- Sig.c_new
    withForeignPtr (tensor t) (tar `Sig.c_resizeAs`)
    withForeignPtr (tensor t) (`Sig.c_copy` tar)
    asDyn <$> newForeignPtr Sig.p_free tar

  copyByte :: Tensor {-ForeignPtr CTHTensor -} -> IO B.DynTensor {-ForeignPtr B.CTorch.FFI.TH.Byte.Tensor -}
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


