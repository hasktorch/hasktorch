{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.TopK where

import Data.Proxy
import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.TopK as Dynamic
import Torch.Indef.Index

topk
  :: forall d' d n
  .  (Dimensions2 d d', KnownDim n)
  => Tensor d -> Integer -> DimVal -> TopKOrder -> Maybe KeepDim -> IO (Tensor d', IndexTensor '[n])
topk t k d o sorted = do
  let ix :: IndexTensor '[n] = newIx
  r  :: Tensor d' <- new
  Dynamic._topk (asDynamic r, longAsDynamic ix) (asDynamic t) k d o sorted
  pure (r, ix)



