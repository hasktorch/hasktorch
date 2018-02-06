module Torch.Class.Storage.Copy where

import Torch.Class.Internal
import Foreign
import Foreign.C.Types
import THTypes

class StorageCopy t where
  rawCopy    :: t -> IO (Ptr (HsReal t))
  copy       :: t -> IO t
  copyByte   :: t -> IO (Ptr CTHByteStorage)
  copyChar   :: t -> IO (Ptr CTHCharStorage)
  copyShort  :: t -> IO (Ptr CTHShortStorage)
  copyInt    :: t -> IO (Ptr CTHIntStorage)
  copyLong   :: t -> IO (Ptr CTHLongStorage)
  copyFloat  :: t -> IO (Ptr CTHFloatStorage)
  copyDouble :: t -> IO (Ptr CTHDoubleStorage)
  copyHalf   :: t -> IO (Ptr CTHHalfStorage)
