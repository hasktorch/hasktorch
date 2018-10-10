-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Index
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Index operations for a dyanmic tensor.
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.Index
  ( _indexCopy
  , _indexAdd
  , _indexFill
  , _indexSelect
  , _take
  , _put
  ) where

import Foreign
import Foreign.Ptr
import Torch.Sig.Types
import Control.Monad.Managed
import qualified Torch.Sig.Types          as Sig
import qualified Torch.Sig.Types.Global   as Sig
import qualified Torch.Sig.Tensor.Index   as Sig

import Torch.Indef.Types

-- | Copies the elements of tensor into the original tensor by selecting the indices in the order given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
_indexCopy :: Dynamic -> Int -> IndexDynamic -> Dynamic -> IO ()
_indexCopy r i ix t = withLift $ Sig.c_indexCopy
  <$> managedState
  <*> managedTensor r
  <*> pure (fromIntegral i)
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t

-- | Accumulate the elements of tensor into the original tensor by adding to the indices in the order given in index. The shape of tensor must exactly match the elements indexed or an error will be thrown.
_indexAdd :: Dynamic -> Int -> IndexDynamic -> Dynamic -> IO ()
_indexAdd r i ix t = withLift $ Sig.c_indexAdd
  <$> managedState
  <*> managedTensor r
  <*> pure (fromIntegral i)
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t

-- | Fills the elements of the original Tensor with value val by selecting the indices in the order given in index.
_indexFill :: Dynamic -> Int -> IndexDynamic -> HsReal -> IO ()
_indexFill r i ix v = withLift $ Sig.c_indexFill
  <$> managedState
  <*> managedTensor r
  <*> pure (fromIntegral i)
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> pure (Sig.hs2cReal v)

-- | Selects the elements of the original Tensor by the index.
_indexSelect :: Dynamic -> Dynamic -> Int -> IndexDynamic -> IO ()
_indexSelect r t i ix = withLift $ Sig.c_indexSelect
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t
  <*> pure (fromIntegral i)
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))

-- | TODO
_take :: Dynamic -> Dynamic -> IndexDynamic -> IO ()
_take r t ix = withLift $ Sig.c_take
  <$> managedState
  <*> managedTensor r
  <*> managedTensor t
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))

-- | TODO
_put :: Dynamic -> IndexDynamic -> Dynamic -> Int -> IO ()
_put r ix t i = withLift $ Sig.c_put
  <$> managedState
  <*> managedTensor r
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t
  <*> pure (fromIntegral i)

-- class GPUTensorIndex Dynamic where
--   _indexCopy_long :: t -> Int -> IndexDynamic t -> t -> IO ()
--   _indexAdd_long :: t -> Int -> IndexDynamic t -> t -> IO ()
--   _indexFill_long :: t -> Int -> IndexDynamic t -> Word -> IO ()
--   _indexSelect_long :: t -> t -> Int -> IndexDynamic t -> IO ()
--   _calculateAdvancedIndexingOffsets :: IndexDynamic t -> t -> Integer -> [IndexTensor t] -> IO ()
