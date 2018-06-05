-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Masked
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Operations using a mask tensor to filter which elements will be used.
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Masked where

import Foreign
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global  as Sig
import qualified Torch.Sig.Tensor.Masked as Sig

import Torch.Indef.Types

-- | fill a dynamic tensor with a value, filtered by a boolean mask tensor
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_maskedFill :: Dynamic -> MaskDynamic -> HsReal -> IO ()
_maskedFill d m v = withDynamicState d $ \s' t' ->
  withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
    Sig.c_maskedFill s' t' m' (hs2cReal v)

-- | copy a dynamic tensor with a value, filtered by a boolean mask tensor
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_maskedCopy :: Dynamic -> MaskDynamic -> Dynamic -> IO ()
_maskedCopy t m f = withDynamicState t $ \s' t' ->
  withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
    withForeignPtr (Sig.ctensor f) $ \f' ->
      Sig.c_maskedCopy s' t' m' f'

-- | select a dynamic tensor with a value, filtered by a boolean mask tensor
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_maskedSelect :: Dynamic -> Dynamic -> MaskDynamic -> IO ()
_maskedSelect t sel m = withDynamicState t $ \s' t' ->
  withForeignPtr (Sig.ctensor sel) $ \sel' ->
    withForeignPtr (snd $ Sig.byteDynamicState m) $ \m' ->
      Sig.c_maskedSelect s' t' sel' m'


-- class GPUTensorMasked t where
--   maskedFillByte   :: t -> MaskDynamic t -> HsReal t -> io ()
--   maskedCopyByte   :: t -> MaskDynamic t -> t -> io ()
--   maskedSelectByte :: t -> t -> MaskDynamic t -> io ()

