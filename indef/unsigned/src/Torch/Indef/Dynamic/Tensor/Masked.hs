module Torch.Indef.Dynamic.Tensor.Masked where

import Foreign
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Sig.Tensor.Masked   as Sig
import qualified Torch.Class.Tensor.Masked as Class

import Torch.Indef.Types

instance Class.TensorMasked Dynamic where
  maskedFill_ :: Dynamic -> MaskTensor -> HsReal -> IO ()
  maskedFill_ d m v = withDynamicState d $ \s' t' ->
    withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
      Sig.c_maskedFill s' t' m' (hs2cReal v)

  maskedCopy_   :: Dynamic -> MaskTensor -> Dynamic -> IO ()
  maskedCopy_ t m f = withDynamicState t $ \s' t' ->
    withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
      withForeignPtr (Sig.ctensor f) $ \f' ->
        Sig.c_maskedCopy s' t' m' f'


  maskedSelect_ :: Dynamic -> Dynamic -> MaskTensor -> IO ()
  maskedSelect_ t sel m = withDynamicState t $ \s' t' ->
    withForeignPtr (Sig.ctensor sel) $ \sel' ->
      withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
        Sig.c_maskedSelect s' t' sel' m'


-- class GPUTensorMasked t where
--   maskedFillByte   :: t -> MaskTensor t -> HsReal t -> io ()
--   maskedCopyByte   :: t -> MaskTensor t -> t -> io ()
--   maskedSelectByte :: t -> t -> MaskTensor t -> io ()

