{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Copy where

import Control.Monad ((>=>))
import qualified Torch.Class.C.Tensor.Copy as CCall

import qualified Torch.Core.LongTensor.Dynamic   as L
import qualified Torch.Core.FloatTensor.Dynamic  as F
import qualified Torch.Core.ByteTensor.Dynamic   as B
-- import qualified Torch.Core.CharTensor.Dynamic   as C
import qualified Torch.Core.ShortTensor.Dynamic  as S
import qualified Torch.Core.IntTensor.Dynamic    as I
import qualified Torch.Core.DoubleTensor.Dynamic as D
-- import qualified Torch.Core.HalfTensor.Dynamic   as H

class CCall.TensorCopy t => UserTensorCopy t where
  copy :: t -> IO t
  copy = CCall.copy

  copyByte :: t -> IO B.Tensor
  copyByte = CCall.copyByte >=> B.asTensor
  -- copyChar   :: t -> IO C.Tensor

  copyShort :: t -> IO S.Tensor
  copyShort = CCall.copyShort >=> S.asTensor

  copyInt :: t -> IO I.Tensor
  copyInt = CCall.copyInt >=> I.asTensor

  copyLong :: t -> IO L.Tensor
  copyLong = CCall.copyLong >=> L.asTensor

  copyFloat :: t -> IO F.Tensor
  copyFloat = CCall.copyFloat >=> F.asTensor

  copyDouble :: t -> IO D.Tensor
  copyDouble = CCall.copyDouble >=> D.asTensor

  --copyHalf   :: t -> IO H.Tensor

instance UserTensorCopy B.Tensor where
instance UserTensorCopy S.Tensor where
instance UserTensorCopy I.Tensor where
instance UserTensorCopy L.Tensor where
instance UserTensorCopy F.Tensor where
instance UserTensorCopy D.Tensor where

