{-# LANGUAGE CPP #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
module Torch.Class.Tensor.Copy where

import Torch.Types.TH
import Foreign (Ptr)
import qualified Torch.Types.TH.Byte    as B
-- import qualified Torch.Types.TH.Short   as S
-- import qualified Torch.Types.TH.Char    as C
-- import qualified Torch.Types.TH.Int     as I
-- import qualified Torch.Types.TH.Long    as L
-- import qualified Torch.Types.TH.Float   as F
-- import qualified Torch.Types.TH.Double  as D

#if CUDA
import qualified Torch.Types.THC as Cuda
#endif

class TensorCopy t where
  copy       :: t -> IO t
  copyByte   :: t -> IO ByteDynamic
  copyChar   :: t -> IO CharDynamic
  copyShort  :: t -> IO ShortDynamic
  copyInt    :: t -> IO IntDynamic
  copyLong   :: t -> IO LongDynamic
  copyDouble :: t -> IO DoubleDynamic
  copyFloat  :: t -> IO FloatDynamic
  -- FIXME: reintroduce Half
  -- copyHalf   :: t -> io H.Dynamic

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
