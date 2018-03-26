{-# LANGUAGE CPP #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
module Torch.Class.Tensor.Copy where

import Torch.Types.TH
import Foreign (Ptr)
import qualified Torch.Types.TH.Byte    as B
import qualified Torch.Types.TH.Short   as S
import qualified Torch.Types.TH.Char    as C
import qualified Torch.Types.TH.Int     as I
import qualified Torch.Types.TH.Long    as L
import qualified Torch.Types.TH.Float   as F
import qualified Torch.Types.TH.Double  as D

#if CUDA
import qualified Torch.Types.THC.Byte    as CudaB
import qualified Torch.Types.THC.Char    as CudaC
import qualified Torch.Types.THC.Short   as CudaS
import qualified Torch.Types.THC.Int     as CudaI
import qualified Torch.Types.THC.Long    as CudaL
import qualified Torch.Types.THC.Double  as CudaD
#endif

class TensorCopy t where
  copy       :: t -> io t
  copyByte   :: t -> io B.DynTensor
  copyChar   :: t -> io C.DynTensor
  copyShort  :: t -> io S.DynTensor
  copyInt    :: t -> io I.DynTensor
  copyLong   :: t -> io L.DynTensor
  -- FIXME: reintroduce Half
  -- copyHalf   :: t -> io H.DynTensor
  copyFloat  :: t -> io F.DynTensor
  copyDouble :: t -> io D.DynTensor

#if CUDA
class GPUTensorCopy gpu cpu | gpu -> cpu where
  copyCuda             :: gpu -> io gpu
  copyIgnoringOverlaps :: gpu -> io gpu

  copyCudaByte    :: gpu -> io CudaB.DynTensor
  copyCudaChar    :: gpu -> io CudaC.DynTensor
  copyCudaShort   :: gpu -> io CudaS.DynTensor
  copyCudaInt     :: gpu -> io CudaI.DynTensor
  copyCudaLong    :: gpu -> io CudaL.DynTensor
  copyCudaDouble  :: gpu -> io CudaD.DynTensor

  copyCPU         :: gpu -> io cpu
  copyAsyncCPU    :: gpu -> io cpu

  thCopyCuda      :: cpu -> io gpu
  thCopyAsyncCuda :: cpu -> io gpu
#endif
