{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Indef.Mask
  ( newMask
  , newMaskDyn
  , allOf
  ) where

import Foreign
import Foreign.Ptr
import Data.Proxy
import Data.List
import Control.Monad
import System.IO.Unsafe

import Torch.Dimensions
import Torch.Sig.Types.Global
import Torch.Indef.Internal
import Control.Monad.Managed as X

import Torch.Sig.State as Sig
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Mask.Tensor as MaskSig
import qualified Torch.Sig.Mask.MathReduce as MaskSig
import qualified Torch.Sig.Mask.TensorFree as MaskSig

newMask :: forall d . Dimensions d => MaskTensor d
newMask = byteAsStatic $ newMaskDyn (dims :: Dims d)

newMaskDyn :: Dims (d::[Nat]) -> MaskDynamic
newMaskDyn d = unsafeDupablePerformIO $ do
  s <- Sig.newCState
  bytePtr <- case fromIntegral <$> listDims d of
    []           -> MaskSig.c_newWithSize1d s 1
    [x]          -> MaskSig.c_newWithSize1d s x
    [x, y]       -> MaskSig.c_newWithSize2d s x y
    [x, y, z]    -> MaskSig.c_newWithSize3d s x y z
    [x, y, z, q] -> MaskSig.c_newWithSize4d s x y z q
    _ -> error "FIXME: can't build masks of this size yet"

  byteDynamic
    <$> Sig.manageState s
    <*> newForeignPtrEnv MaskSig.p_free s bytePtr

class IsMask t where
  allOf :: t -> Bool
  -- anyOf :: t -> Bool

instance IsMask MaskDynamic where
  allOf t = unsafeDupablePerformIO $ flip X.with pure $ do
    s' <- managed $ withForeignPtr s
    t' <- managed $ withForeignPtr fp

    liftIO $ do
      ds <- MaskSig.c_nDimension s' t'
      prod <- foldM (\acc d -> (acc *) <$> MaskSig.c_size s' t' (fromIntegral d)) 1 [0..ds-1]
      tot  <- MaskSig.c_sumall s' t'
      pure $ tot == fromIntegral prod

    where
      (s, fp) = byteDynamicState t

instance IsMask (MaskTensor d) where
  allOf = allOf . byteAsDynamic
