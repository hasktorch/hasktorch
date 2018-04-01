{-# LANGUAGE CPP #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
module Torch.Class.Tensor.Copy.Static where

import Torch.Types.TH
import Torch.Dimensions

#if CUDA
import qualified Torch.Types.THC as Cuda
#endif

class TensorCopy t where
  copy       :: Dimensions d => t d -> IO (t d)
  copyByte   :: Dimensions d => t d -> IO (ByteTensor d)
  copyChar   :: Dimensions d => t d -> IO (CharTensor d)
  copyShort  :: Dimensions d => t d -> IO (ShortTensor d)
  copyInt    :: Dimensions d => t d -> IO (IntTensor d)
  copyLong   :: Dimensions d => t d -> IO (LongTensor d)
  copyDouble :: Dimensions d => t d -> IO (DoubleTensor d)
  copyFloat  :: Dimensions d => t d -> IO (FloatTensor d)
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
