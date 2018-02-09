module Torch.Core.Storage.Copy
  ( copy
  , copyByte
  -- , copyChar
  -- , copyShort
  -- , copyInt
  -- , copyLong
  -- , copyHalf
  -- , copyFloat
  -- , copyDouble

  , Class.StorageCopy
  ) where

import Control.Monad ((>=>))
import qualified Torch.Core.ByteStorage as B
-- import qualified Torch.Core.ShortStorage as S
-- import qualified Torch.Core.IntStorage as I
-- import qualified Torch.Core.LongStorage as L
-- import qualified Torch.Core.FloatStorage as F
-- import qualified Torch.Core.DoubleStorage as D

import qualified Torch.Class.C.Storage.Copy as Class

copy :: Class.StorageCopy t => t -> IO t
copy = Class.copy

copyByte :: Class.StorageCopy t => t -> IO B.Storage
copyByte = Class.copyByte >=> B.asStorage

-- copyChar   :: t -> IO (Ptr CTHCharTensor)

-- copyShort :: t -> IO S.Storage
-- copyShort = C.copyShort >=> S.asStorage
--
-- copyInt :: t -> IO I.Storage
-- copyInt = C.copyInt >=> I.asStorage
--
-- copyLong :: t -> IO L.Storage
-- copyLong = C.copyLong >=> L.asStorage
--
--
-- copyFloat  :: t -> IO F.Storage
-- copyFloat = C.copyFloat >=> F.asStorage
--
-- --copyHalf   :: t -> IO (Ptr CTHHalfTensor)
--
-- copyDouble :: t -> IO D.Storage
-- copyDouble = C.copyDouble >=> D.asStorage


