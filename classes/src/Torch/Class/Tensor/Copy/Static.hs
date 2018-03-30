{-# LANGUAGE CPP #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
module Torch.Class.Tensor.Copy.Static where

import Torch.Types.TH
import Torch.Dimensions

#if CUDA
import qualified Torch.Types.THC as Cuda
#endif

class Dimensions d => TensorCopy t d where
  copy       :: t d -> IO (t d)
  copyByte   :: t d -> IO (ByteTensor d)
  copyChar   :: t d -> IO (CharTensor d)
  copyShort  :: t d -> IO (ShortTensor d)
  copyInt    :: t d -> IO (IntTensor d)
  copyLong   :: t d -> IO (LongTensor d)
  copyDouble :: t d -> IO (DoubleTensor d)
  copyFloat  :: t d -> IO (FloatTensor d)
  -- FIXME: reintroduce Half
  -- copyHalf   :: t -> io H.Dynamic

{-
#if CUDA
class GPUTensorCopy gpu cpu | gpu -> cpu where
  copyCuda             :: gpu -> io gpu
  copyIgnoringOverlaps :: gpu -> io gpu

  copyCudaByte    :: gpu -> IO Cuda.ByteDynamic
  copyCudaChar    :: gpu -> IO Cuda.CharDynamic
  copyCudaShort   :: gpu -> IO Cuda.ShortDynamic
  copyCudaInt     :: gpu -> IO Cuda.IntDynamic
  copyCudaLong    :: gpu -> IO Cuda.LongDynamic
  copyCudaDouble  :: gpu -> IO Cuda.DoubleDynamic

  copyCPU         :: gpu -> IO cpu
  copyAsyncCPU    :: gpu -> IO cpu

  thCopyCuda      :: cpu -> IO gpu
  thCopyAsyncCuda :: cpu -> IO gpu
#endif
  -}
