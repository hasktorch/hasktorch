module Torch.Distributions.Distribution
  ( Scale,
    Distribution (..),
    stddev,
    perplexity,
    logitsToProbs,
    clampProbs,
    probsToLogits,
    extendedShape,
  )
where

import Torch.Distributions.Constraints
import qualified Torch.Functional as F
import qualified Torch.Tensor as D
import Torch.TensorFactories (ones, onesLike)

data Scale = Probs | Logits

class Distribution a where
  batchShape :: a -> [Int]
  eventShape :: a -> [Int]
  expand :: a -> [Int] -> a
  support :: a -> Constraint
  mean :: a -> D.Tensor
  variance :: a -> D.Tensor
  sample :: a -> [Int] -> IO D.Tensor
  logProb :: a -> D.Tensor -> D.Tensor
  entropy :: a -> D.Tensor
  enumerateSupport :: a -> Bool -> D.Tensor -- (expand=True)

stddev :: (Distribution a) => a -> D.Tensor -- 'D.Float
stddev = F.sqrt . variance

-- Tensor device 'D.Float '[batchShape]
perplexity :: (Distribution a) => a -> D.Tensor
perplexity = F.exp . entropy

-- | Converts a tensor of logits into probabilities. Note that for the
-- | binary case, each value denotes log odds, whereas for the
-- | multi-dimensional case, the values along the last dimension denote
-- | the log probabilities (possibly unnormalized) of the events.
logitsToProbs :: Bool -> D.Tensor -> D.Tensor -- isBinary=False
logitsToProbs True = F.sigmoid
logitsToProbs False = F.softmax (F.Dim $ -1)

clampProbs :: D.Tensor -> D.Tensor
clampProbs probs =
  F.clamp eps (1.0 - eps) probs
  where
    eps = 0.000001 -- torch.finfo(probs.dtype).eps

-- | Converts a tensor of probabilities into logits. For the binary case,
-- | this denotes the probability of occurrence of the event indexed by `1`.
-- | For the multi-dimensional case, the values along the last dimension
-- | denote the probabilities of occurrence of each of the events.
probsToLogits :: Bool -> D.Tensor -> D.Tensor -- isBinary=False
probsToLogits isBinary probs =
  if isBinary
    then F.log10 psClamped `F.sub` F.log1p (F.mulScalar (-1.0 :: Float) psClamped)
    else F.log10 psClamped
  where
    psClamped = clampProbs probs

-- | Returns the size of the sample returned by the distribution, given
-- | a `sampleShape`. Note, that the batch and event shapes of a distribution
-- | instance are fixed at the time of construction. If this is empty, the
-- | returned shape is upcast to (1,).
-- | Args:
-- |     sampleShape (torch.Size): the size of the sample to be drawn.
extendedShape :: (Distribution a) => a -> [Int] -> [Int]
extendedShape d sampleShape =
  sampleShape <> batchShape d <> eventShape d
