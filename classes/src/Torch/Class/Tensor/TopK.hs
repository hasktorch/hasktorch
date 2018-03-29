{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.Tensor.TopK where

import Torch.Class.Types
import Torch.Dimensions
import Torch.Class.Tensor

-- https://github.com/torch/torch7/blob/75a86469aa9e2f5f04e11895b269ec22eb0e4687/lib/TH/generic/THTensorMath.c#L2545
data TopKOrder = KAscending | KNone | KDescending
  deriving (Eq, Show, Ord, Enum, Bounded)

class TensorTopK t where
  topk_ :: (t, IndexTensor t) -> t -> Integer -> DimVal -> TopKOrder -> Bool -> IO ()

topk
  :: forall t d' d
  .  (TensorTopK (AsDynamic (t d)))
  => (Dimensions d, Dimensions d')
  => (Static2 (t d) (t d'))
  => t d -> Integer -> DimVal -> TopKOrder -> Bool -> IO (t d', IndexTensor (t d'))
topk t k d o sorted = do
  ix :: IndexTensor (t d) <- new (dim :: Dim d')
  r  :: AsDynamic (t d) <- new (dim :: Dim d')
  topk_ (r, ix) (asDynamic t) k d o sorted
  pure (asStatic r, ix)



