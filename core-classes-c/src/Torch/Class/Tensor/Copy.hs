module Torch.Class.Tensor.Copy where

import THTypes
import Foreign (Ptr)

class TensorCopy t where
  copy       :: t -> IO t
  copyByte   :: t -> IO (Ptr CTHByteTensor)
  copyChar   :: t -> IO (Ptr CTHCharTensor)
  copyShort  :: t -> IO (Ptr CTHShortTensor)
  copyInt    :: t -> IO (Ptr CTHIntTensor)
  copyLong   :: t -> IO (Ptr CTHLongTensor)
  copyFloat  :: t -> IO (Ptr CTHFloatTensor)
  copyDouble :: t -> IO (Ptr CTHDoubleTensor)
  copyHalf   :: t -> IO (Ptr CTHHalfTensor)
