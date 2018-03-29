{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Indef.Dynamic.Tensor.Copy where

import Foreign
import Foreign.C.Types
import Data.List (intercalate)
import Control.Exception.Safe (throwString)
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Tensor as Sig
import qualified Torch.Sig.Tensor.Copy as Sig
import qualified Torch.Class.Tensor.Copy as Class
import qualified Torch.Class.Tensor as Class

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
  -> Dynamic -> IO b
copyType newPtr fin builder cfun t = withDynamicState t $ \s' t' -> do
  target <- newPtr
  throwString $ intercalate ""
    [ "'hasktorch-indef-unsigned:Torch.Indef.Tensor.Dynamic.Copy.copyType':"
    , "must resize the target tensor before continuing"
    ]
  -- Sig.c_resizeAs s' target t'       -- << THIS NEEDS TO BE REMAPPED TO TENSORLONG SIZES
  cfun s' t' target

  builder
    <$> (TH.newCState >>= TH.manageState)
    <*> newForeignPtr fin target


instance Class.TensorCopy Dynamic where
  copy :: Dynamic -> IO Dynamic
  copy t = withDynamicState t $ \s' t' -> do
    target <- Sig.c_new s'
    Sig.c_resizeAs s' target t'
    Sig.c_copy s' t' target
    mkDynamic s' target

  copyByte   = copyType B.c_new_ B.p_free   TH.byteDynamic Sig.c_copyByte
  copyChar   = copyType C.c_new_ C.p_free   TH.charDynamic Sig.c_copyChar
  copyShort  = copyType S.c_new_ S.p_free  TH.shortDynamic Sig.c_copyShort
  copyInt    = copyType I.c_new_ I.p_free    TH.intDynamic Sig.c_copyInt
  copyLong   = copyType L.c_new_ L.p_free   TH.longDynamic Sig.c_copyLong
  copyFloat  = copyType F.c_new_ F.p_free  TH.floatDynamic Sig.c_copyFloat
  copyDouble = copyType D.c_new_ D.p_free TH.doubleDynamic Sig.c_copyDouble


