{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Distributions.Bernoulli
  ( Bernoulli (..),
    fromProbs,
    fromLogits,
  )
where

import qualified Torch.DType as D
import qualified Torch.Distributions.Constraints as Constraints
import Torch.Distributions.Distribution
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as I
import Torch.Scalar
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import Torch.TensorOptions
import Torch.Typed.Functional (reductionVal)

data Bernoulli = Bernoulli
  { probs :: D.Tensor,
    logits :: D.Tensor
  }
  deriving (Show)

instance Distribution Bernoulli where
  batchShape d = []
  eventShape _d = []
  expand d = fromProbs . F.expand (probs d) False
  support d = Constraints.boolean
  mean = probs
  variance d = p `F.mul` (D.onesLike p `F.sub` p)
    where
      p = probs d
  sample d = D.bernoulliIO' . F.expand (probs d) False . extendedShape d
  logProb d value = F.mulScalar (-1 :: Int) (bce' (logits d) value)
  entropy d = bce' (logits d) $ probs d
  enumerateSupport d doExpand =
    (if doExpand then \t -> F.expand t False ([-1] <> batchShape d) else id) values
    where
      values = D.reshape ([-1] <> replicate (length $ batchShape d) 1) $ D.asTensor [0.0, 1.0 :: Float]

bce' :: D.Tensor -> D.Tensor -> D.Tensor
bce' logits probs =
  I.binary_cross_entropy_with_logits
    logits
    probs
    (D.onesLike logits)
    (D.ones [D.size (-1) logits] D.float_opts)
    $ reductionVal @(F.ReduceNone)

fromProbs :: D.Tensor -> Bernoulli
fromProbs probs = Bernoulli probs $ probsToLogits False probs

fromLogits :: D.Tensor -> Bernoulli
fromLogits logits = Bernoulli (probsToLogits False logits) logits
