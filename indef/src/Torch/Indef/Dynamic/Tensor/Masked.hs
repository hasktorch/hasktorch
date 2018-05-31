module Torch.Indef.Dynamic.Tensor.Masked where

import Foreign
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global  as Sig
import qualified Torch.Sig.Tensor.Masked as Sig

import Torch.Indef.Types

_maskedFill :: Dynamic -> MaskDynamic -> HsReal -> IO ()
_maskedFill d m v = withDynamicState d $ \s' t' ->
  withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
    Sig.c_maskedFill s' t' m' (hs2cReal v)

_maskedCopy   :: Dynamic -> MaskDynamic -> Dynamic -> IO ()
_maskedCopy t m f = withDynamicState t $ \s' t' ->
  withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
    withForeignPtr (Sig.ctensor f) $ \f' ->
      Sig.c_maskedCopy s' t' m' f'


_maskedSelect :: Dynamic -> Dynamic -> MaskDynamic -> IO ()
_maskedSelect t sel m = withDynamicState t $ \s' t' ->
  withForeignPtr (Sig.ctensor sel) $ \sel' ->
    withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
      Sig.c_maskedSelect s' t' sel' m'


-- class GPUTensorMasked t where
--   maskedFillByte   :: t -> MaskDynamic t -> HsReal t -> io ()
--   maskedCopyByte   :: t -> MaskDynamic t -> t -> io ()
--   maskedSelectByte :: t -> t -> MaskDynamic t -> io ()

