module Torch.Distributions.Categorical (
    Categorical(..),
    fromProbs,
    fromLogits,
) where

import qualified Torch.Functional.Internal as I
import qualified Torch.Tensor as D
import qualified Torch.DType as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functional as F
import qualified Torch.Distributions.Constraints as Constraints
import Torch.Distributions.Distribution

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
-- | See also: :func:`torch.multinomial`
-- | Example::
-- |     >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
-- |     >>> m.sample()  # equal probability of 0, 1, 2, 3
-- |     tensor(3)
-- | Args:
-- |     probs (Tensor): event probabilities
-- |     logits (Tensor): event log-odds
data Categorical = Categorical {
    probs :: D.Tensor,
    logits :: D.Tensor
} deriving (Show)
instance Distribution Categorical where
    batch_shape d =
        if D.numel (probs d) > 1
            then init (D.shape $ probs d)
            else []
    event_shape _d = []
    expand d batch_shape' = fromProbs $ F.expand (probs d) False (param_shape d)
        where param_shape d' = batch_shape' <> [num_events d']
    support d = Constraints.integerInterval 0 $ (num_events d) - 1
    mean d = F.divScalar (D.ones (extended_shape d []) D.float_opts) (0.0 :: Float)  -- all NaN
    variance d = F.divScalar (D.ones (extended_shape d []) D.float_opts) (0.0 :: Float)  -- all NaN
    sample d sample_shape = do
        let probs_2d = D.reshape [-1, (num_events d)] $ probs d
        samples_2d <- F.transpose2D <$> F.multinomial_tlb probs_2d (product sample_shape) True
        return $ D.reshape (extended_shape d sample_shape) samples_2d
    log_prob d value = let
        value' = I.unsqueeze (F.toDType D.Int64 value) (-1 :: Int)
        value'' = D.select value' (-1) 0
        in F.squeezeDim (-1) $ I.gather (logits d) (-1 :: Int) value'' False
    entropy d = F.mulScalar (F.sumDim (-1) p_log_p) (-1.0 :: Float)
            where p_log_p = logits d `F.mul` probs d
    enumerate_support d do_expand = 
        (if do_expand then \t -> F.expand t False ([-1] <> batch_shape d) else id) values
        where
            values = D.reshape ([-1] <> replicate (length $ batch_shape d) 1) $ D.asTensor [0.0, 1.0 :: Float]

num_events :: Categorical -> Int
num_events (Categorical ps _logits) = D.size ps (-1)

fromProbs :: D.Tensor -> Categorical
fromProbs ps = Categorical ps $ probs_to_logits False ps

fromLogits :: D.Tensor -> Categorical
fromLogits logits' = Categorical (probs_to_logits False logits') logits'
