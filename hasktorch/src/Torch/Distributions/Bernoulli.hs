{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Distributions.Bernoulli (
    Bernoulli(..),
    fromProbs,
    fromLogits,
) where

import Torch.Typed.Functional (reductionVal)
import qualified Torch.Functional.Internal as I
import Torch.TensorOptions
import qualified Torch.Tensor as D
import qualified Torch.DType as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functional as F
import Torch.Scalar
import qualified Torch.Distributions.Constraints as Constraints
import Torch.Distributions.Distribution

data Bernoulli = Bernoulli {
    probs :: D.Tensor,
    logits :: D.Tensor
} deriving (Show)
instance Distribution Bernoulli where
    batch_shape d = []
    event_shape d = []
    expand d = fromProbs . F.expand (probs d) False
    support d = Constraints.boolean
    mean = probs
    variance d = p `F.mul` (D.onesLike p `F.sub` p)
            where p = probs d
    sample d = F.bernoulli_t . F.expand (probs d) False . extended_shape d
    log_prob d value = F.mulScalar (bce' (logits d) value) (-1 :: Int)
    entropy d = bce' (logits d) $ probs d
    enumerate_support d do_expand = 
        (if do_expand then \t -> F.expand t False ([-1] <> batch_shape d) else id) values
        where
            values = D.reshape ([-1] <> replicate (length $ batch_shape d) 1) $ D.asTensor [0.0, 1.0 :: Float]

bce' :: D.Tensor -> D.Tensor -> D.Tensor
bce' logits probs = I.binary_cross_entropy_with_logits logits probs
                        (D.onesLike logits) (D.ones [D.size logits (-1)] D.float_opts)
                        $ reductionVal @(F.ReduceNone)

fromProbs :: D.Tensor -> Bernoulli
fromProbs probs = Bernoulli probs $ probs_to_logits False probs

fromLogits :: D.Tensor -> Bernoulli
fromLogits logits = Bernoulli (probs_to_logits False logits) logits
