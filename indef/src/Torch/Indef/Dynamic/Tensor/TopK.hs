-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.TopK
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor.TopK where

import Foreign
import Control.Monad.Managed
import Torch.Indef.Types
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Sig.Tensor.TopK as Sig
import qualified Torch.Indef.Index as Ix

-- | returns all @k@ smallest elements in a tensor over a given dimension, including their indices, in unsorted order.
_topk
  :: (Dynamic, IndexDynamic)
  -> Dynamic
  -> Integer
  -> Word           -- ^ dimension to operate on
  -> TopKOrder
  -> Maybe KeepDim
  -> IO ()
_topk (r, ix) t1 l i0 o sorted = withLift $ Sig.c_topk
  <$> managedState
  <*> managedTensor r
  <*> managed (withForeignPtr (snd $ Sig.longDynamicState ix))
  <*> managedTensor t1
  <*> pure (fromIntegral l)
  <*> pure (fromIntegral i0)
  <*> pure (fromIntegral $ fromEnum o)
  <*> pure (fromKeepDim sorted)

