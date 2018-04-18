{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
module Torch.Class.Storage.Copy where

import Torch.Class.Types
import Foreign (Ptr)
import Torch.Types.TH
import Control.Monad.IO.Class

import Torch.Types.TH

class StorageCopy t where
  rawCopy    :: t -> IO [HsReal t]
  copy       :: t -> IO t
  copyByte   :: t -> IO ByteStorage
  copyChar   :: t -> IO CharStorage
  copyShort  :: t -> IO ShortStorage
  copyInt    :: t -> IO IntStorage
  copyLong   :: t -> IO LongStorage
  copyFloat  :: t -> IO FloatStorage
  copyDouble :: t -> IO DoubleStorage
  -- FIXME: reintroduce half
  -- copyHalf   :: t -> IO HalfStorage

class GPUStorageCopy gpu cpu | gpu -> cpu where
  thCopyCuda :: cpu -> IO gpu
  copyCuda   :: gpu -> IO gpu
  copyCPU    :: gpu -> IO cpu

