module Torch.Core.Storage.Copy where

import Control.Monad ((>=>))
import qualified Torch.Core.ByteStorage as B
import qualified Torch.Core.ShortStorage as S
import qualified Torch.Core.IntStorage as I
import qualified Torch.Core.LongStorage as L

import qualified Torch.Class.C.Storage.Copy as C

class C.StorageCopy t => UserStorageCopy t where
  copy :: t -> IO t
  copy = C.copy

  copyByte :: t -> IO B.Storage
  copyByte = C.copyByte >=> B.asStorage
  -- copyChar   :: t -> IO (Ptr CTHCharTensor)

  copyShort :: t -> IO S.Storage
  copyShort = C.copyShort >=> S.asStorage

  copyInt :: t -> IO I.Storage
  copyInt = C.copyInt >=> I.asStorage

  copyLong :: t -> IO L.Storage
  copyLong = C.copyLong >=> L.asStorage


  --copyFloat  :: t -> IO (Ptr CTHFloatTensor)
  --copyHalf   :: t -> IO (Ptr CTHHalfTensor)
  --copyDouble :: t -> IO (Ptr CTHDoubleTensor)

instance UserStorageCopy B.Storage where
instance UserStorageCopy S.Storage where
instance UserStorageCopy I.Storage where
instance UserStorageCopy L.Storage where
