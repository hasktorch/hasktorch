module Torch.Distributions.Categorical
  ( Categorical (..),
    fromProbs,
    fromLogits,
  )
where

import qualified Torch.DType as D
import qualified Torch.Distributions.Constraints as Constraints
import Torch.Distributions.Distribution
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as I
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D

-- | Creates a categorical distribution parameterized by either :attr:`probs` or
-- | :attr:`logits` (but not both).
-- | .. note::
-- |     It is equivalent to the distribution that :func:`torch.multinomial`
-- |     samples from.
-- | Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.
-- | If :attr:`probs` is 1D with length-`K`, each element is the relative
-- | probability of sampling the class at that index.
-- | If :attr:`probs` is 2D, it is treated as a batch of relative probability
-- | vectors.
-- | .. note:: :attr:`probs` must be non-negative, finite and have a non-zero sum,
-- |             and it will be normalized to sum to 1.
-- | See also: `torch.multinomial`
-- | Example::
-- |     >>> m = Categorical.fromProbs $ D.asTensor [ 0.25, 0.25, 0.25, 0.25 ]
-- |     >>> Distribution.sample m  -- equal probability of 0, 1, 2, 3
-- |     tensor(3)
data Categorical = Categorical
  { probs :: D.Tensor,
    logits :: D.Tensor
  }
  deriving (Show)

instance Distribution Categorical where
  batchShape d =
    if D.numel (probs d) > 1
      then init (D.shape $ probs d)
      else []
  eventShape _d = []
  expand d batchShape' = fromProbs $ F.expand (probs d) False (paramShape d)
    where
      paramShape d' = batchShape' <> [numEvents d']
  support d = Constraints.integerInterval 0 $ (numEvents d) - 1
  mean d = F.divScalar (0.0 :: Float) (D.ones (extendedShape d []) D.float_opts) -- all NaN
  variance d = F.divScalar (0.0 :: Float) (D.ones (extendedShape d []) D.float_opts) -- all NaN
  sample d sampleShape = do
    let probs2d = D.reshape [-1, (numEvents d)] $ probs d
    samples2d <- F.transpose2D <$> D.multinomialIO probs2d (product sampleShape) True
    return $ D.reshape (extendedShape d sampleShape) samples2d
  logProb d value =
    let value' = I.unsqueeze (F.toDType D.Int64 value) (-1 :: Int)
        value'' = D.select (-1) 0 value'
     in F.squeezeDim (-1) $ I.gather (logits d) (-1 :: Int) value'' False
  entropy d = F.mulScalar (-1.0 :: Float) (F.sumDim (F.Dim $ -1) F.RemoveDim (D.dtype pLogP) pLogP)
    where
      pLogP = logits d `F.mul` probs d
  enumerateSupport d doExpand =
    (if doExpand then \t -> F.expand t False ([-1] <> batchShape d) else id) values
    where
      values = D.reshape ([-1] <> replicate (length $ batchShape d) 1) $ D.asTensor [0.0, 1.0 :: Float]

numEvents :: Categorical -> Int
numEvents (Categorical ps _logits) = D.size (-1) ps

fromProbs :: D.Tensor -> Categorical
fromProbs ps = Categorical ps $ probsToLogits False ps

fromLogits :: D.Tensor -> Categorical
fromLogits logits' = Categorical (logitsToProbs False logits') logits'
