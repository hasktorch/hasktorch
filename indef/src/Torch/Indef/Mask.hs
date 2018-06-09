-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Mask
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Redundant version of @Torch.Indef.{Dynamic/Static}.Tensor@ for Byte tensors.
--
-- This comes with the same fixme as 'Torch.Indef.Index':
--
-- FIXME: in the future, there could be a smaller subset of Torch which could
-- be compiled to to keep the code dry. Alternatively, if backpack one day
-- supports recursive indefinites, we could use this feature to possibly remove
-- this package and 'Torch.Indef.Mask'.
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Indef.Mask
  ( newMask
  , newMaskDyn
  , withMask
  , allOf
  ) where

import Foreign
import Foreign.Ptr
import Data.Proxy
import Data.List
import Control.Monad
import System.IO.Unsafe

import Numeric.Dimensions
import Torch.Sig.Types.Global
import Torch.Indef.Internal
import Control.Monad.Managed as X

import Torch.Sig.State as Sig
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Mask.Tensor as MaskSig
import qualified Torch.Sig.Mask.MathReduce as MaskSig
import qualified Torch.Sig.Mask.TensorFree as MaskSig

-- | build a new mask tensor with any known Dimension list.
newMask :: forall d . Dimensions d => MaskTensor d
newMask = byteAsStatic $ newMaskDyn (dims :: Dims d)

-- | build a new dynamic mask tensor with any known Nat list.
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

-- | run a function with access to a dynamic index tensor's raw c-pointer.
withMask :: MaskDynamic -> (Ptr CMaskTensor -> IO x) -> IO x
withMask ix fn = withForeignPtr (snd $ byteDynamicState ix) fn

class IsMask t where
  -- | assert that all of the values of the Byte tensor are true.
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
