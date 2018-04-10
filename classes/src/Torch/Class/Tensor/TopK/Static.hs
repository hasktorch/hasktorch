{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.Tensor.TopK.Static
  ( TopKOrder(..)
  , TensorTopK(..)
  , topk
  ) where

import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor.Static
import Torch.Class.Tensor.TopK (TopKOrder(..))
import qualified Torch.Class.Tensor as Dynamic

class IsTensor t => TensorTopK t where
  _topk
    :: (Dimensions d, Dimensions d')
    => (t d', IndexTensor (t d') d')
    -> t d
    -> Integer
    -> DimVal
    -> TopKOrder
    -> Maybe KeepDim
    -> IO ()

topk
  :: forall t d' d
  .  (TensorTopK t)
  => (Dimensions d, Dimensions d')
  => (Dynamic.IsTensor (IndexTensor (t d') d'))
  => t d -> Integer -> DimVal -> TopKOrder -> Maybe KeepDim -> IO (t d', IndexTensor (t d') d')
topk t k d o sorted = do
  ix :: IndexTensor (t d') d' <- Dynamic.new (dim :: Dim d')
  r  :: t d' <- new
  _topk (r, ix) t k d o sorted
  pure (r, ix)



