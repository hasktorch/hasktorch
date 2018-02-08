module Torch.Core.Storage.Copy where

import qualified Torch.Core.ByteStorage as B
import qualified Torch.Core.ShortStorage as S

import qualified Torch.Class.Storage.Copy as C

class C.StorageCopy t => UserStorageCopy t where
  copy :: t -> IO t
  copy = C.copy

  copyByte :: t -> IO B.Storage
  --copyChar   :: t -> IO (Ptr CTHCharTensor)
  copyShort  :: t -> IO S.Storage
  --copyInt    :: t -> IO (Ptr CTHIntTensor)
  --copyLong   :: t -> IO (Ptr CTHLongTensor)
  --copyFloat  :: t -> IO (Ptr CTHFloatTensor)
  --copyDouble :: t -> IO (Ptr CTHDoubleTensor)
  --copyHalf   :: t -> IO (Ptr CTHHalfTensor)

instance UserStorageCopy B.Storage where
  copyByte = copy
  copyShort s = C.copyShort s >>= S.asStorage

instance UserStorageCopy S.Storage where
  copyShort = copy
  copyByte s = C.copyByte s >>= B.asStorage
