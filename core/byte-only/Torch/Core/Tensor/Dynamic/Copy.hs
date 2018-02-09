{-# LANGUAGE InstanceSigs #-}
module Torch.Core.Tensor.Dynamic.Copy
  ( copy
  , copyByte
  -- , copyChar
  -- , copyShort
  -- , copyInt
  -- , copyLong
  -- , copyFloat
  -- , copyDouble
  -- , copyHalf
  , Class.TensorCopy
  ) where

import Control.Monad ((>=>))
import qualified Torch.Class.C.Tensor.Copy as Class

import qualified Torch.Core.ByteTensor.Dynamic   as B
-- -- import qualified Torch.Core.CharTensor.Dynamic   as C
-- import qualified Torch.Core.ShortTensor.Dynamic  as S
-- import qualified Torch.Core.IntTensor.Dynamic    as I
-- import qualified Torch.Core.LongTensor.Dynamic   as L
-- -- import qualified Torch.Core.HalfTensor.Dynamic   as H
-- import qualified Torch.Core.FloatTensor.Dynamic  as F
-- import qualified Torch.Core.DoubleTensor.Dynamic as D

copy :: Class.TensorCopy t => t -> IO t
copy = Class.copy

copyByte :: Class.TensorCopy t => t -> IO B.Tensor
copyByte = Class.copyByte >=> B.asTensor

-- -- copyChar :: Class.TensorCopy t => t -> IO C.Tensor
--
-- copyShort :: Class.TensorCopy t => t -> IO S.Tensor
-- copyShort = Class.copyShort >=> S.asTensor
--
-- copyInt :: Class.TensorCopy t => t -> IO I.Tensor
-- copyInt = Class.copyInt >=> I.asTensor
--
-- copyLong :: Class.TensorCopy t => t -> IO L.Tensor
-- copyLong = Class.copyLong >=> L.asTensor
--
-- copyFloat :: Class.TensorCopy t => t -> IO F.Tensor
-- copyFloat = Class.copyFloat >=> F.asTensor
--
-- copyDouble :: Class.TensorCopy t => t -> IO D.Tensor
-- copyDouble = Class.copyDouble >=> D.asTensor
--
-- --copyHalf :: Class.TensorCopy t => t -> IO H.Tensor

