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

import Foreign hiding (with)
import Control.Monad.Managed
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global  as Sig
import qualified Torch.Sig.Tensor.Masked as Sig

import Torch.Indef.Types

-- | fill a dynamic tensor with a value, filtered by a boolean mask tensor
maskedFill_
  :: Dynamic     -- ^ source tensor to mutate, inplace
  -> MaskDynamic -- ^ mask to fill
  -> HsReal      -- ^ value to fill
  -> IO ()
maskedFill_ d m v = withLift $ Sig.c_maskedFill
  <$> managedState
  <*> managedTensor d
  <*> managed (withForeignPtr (snd $ Sig.byteDynamicState m))
  <*> pure (hs2cReal v)

-- | copy a dynamic tensor with a value, filtered by a boolean mask tensor
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_maskedCopy
  :: Dynamic     -- ^ return tensor to mutate, inplace
  -> MaskDynamic -- ^ mask to copy with
  -> Dynamic     -- ^ source tensor to copy from
  -> IO ()
_maskedCopy t m f = withLift $ Sig.c_maskedCopy
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (snd $ Sig.byteDynamicState m))
  <*> managedTensor f

-- | select a dynamic tensor with a value, filtered by a boolean mask tensor
--
-- C-Style: In the classic Torch C-style, the first argument is treated as the return type and is mutated in-place.
_maskedSelect
  :: Dynamic     -- ^ return tensor to mutate, inplace
  -> Dynamic     -- ^ source tensor to select from
  -> MaskDynamic -- ^ mask to select with
  -> IO ()
_maskedSelect t sel m = withLift $ Sig.c_maskedSelect
  <$> managedState
  <*> managedTensor t
  <*> managedTensor sel
  <*> managed (withForeignPtr (snd $ Sig.byteDynamicState m))

-- class GPUTensorMasked t where
--   maskedFillByte   :: t -> MaskDynamic t -> HsReal t -> io ()
--   maskedCopyByte   :: t -> MaskDynamic t -> t -> io ()
--   maskedSelectByte :: t -> t -> MaskDynamic t -> io ()

