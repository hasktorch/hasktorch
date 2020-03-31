module Torch.Distributions.Distribution (
    Scale,
    Distribution(..),
    stddev,
    perplexity,
    logits_to_probs,
    clamp_probs,
    probs_to_logits,
    extended_shape,
) where

import Torch.Typed.Functional
import Torch.Functional.Internal (binary_cross_entropy_with_logits, log)
import Torch.TensorFactories (ones, onesLike)
import Torch.TensorOptions
import qualified Torch.Tensor as D
import qualified Torch.DType as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functional as F
import Torch.Scalar
import Torch.Distributions.Constraints

data Scale = Probs | Logits

class Distribution a where
    batch_shape :: a -> [Int]
    event_shape :: a -> [Int]
    expand :: a -> [Int] -> a
    support :: a -> Constraint
    mean :: a -> D.Tensor
    variance :: a -> D.Tensor
    sample :: a -> [Int] -> IO D.Tensor
    log_prob :: a -> D.Tensor -> D.Tensor
    entropy :: a -> D.Tensor
    enumerate_support :: a -> Bool -> D.Tensor -- (expand=True)

stddev :: (Distribution a) => a -> D.Tensor -- 'D.Float
stddev = F.sqrt . variance

-- Tensor device 'D.Float '[batch_shape]
perplexity :: (Distribution a) => a -> D.Tensor
perplexity = F.exp . entropy

-- | Converts a tensor of logits into probabilities. Note that for the
-- | binary case, each value denotes log odds, whereas for the
-- | multi-dimensional case, the values along the last dimension denote
-- | the log probabilities (possibly unnormalized) of the events.
logits_to_probs :: Bool -> D.Tensor -> D.Tensor  -- is_binary=False
logits_to_probs True = F.sigmoid
logits_to_probs False = F.softmax (-1)

clamp_probs :: D.Tensor -> D.Tensor
clamp_probs probs =
    F.clamp eps (1.0 - eps) probs
    where eps = 0.000001 -- torch.finfo(probs.dtype).eps

-- | Converts a tensor of probabilities into logits. For the binary case,
-- | this denotes the probability of occurrence of the event indexed by `1`.
-- | For the multi-dimensional case, the values along the last dimension
-- | denote the probabilities of occurrence of each of the events.
probs_to_logits :: Bool -> D.Tensor -> D.Tensor  -- is_binary=False
probs_to_logits is_binary probs =
    if is_binary
        then F.log10 ps_clamped `F.sub` F.log1p (F.mulScalar ps_clamped (-1.0 :: Float))
        else F.log10 ps_clamped
    where ps_clamped = clamp_probs probs

-- | Returns the size of the sample returned by the distribution, given
-- | a `sample_shape`. Note, that the batch and event shapes of a distribution
-- | instance are fixed at the time of construction. If this is empty, the
-- | returned shape is upcast to (1,).
-- | Args:
-- |     sample_shape (torch.Size): the size of the sample to be drawn.
extended_shape :: (Distribution a) => a -> [Int] -> [Int]
extended_shape d sample_shape =
        sample_shape <> batch_shape d <> event_shape d
