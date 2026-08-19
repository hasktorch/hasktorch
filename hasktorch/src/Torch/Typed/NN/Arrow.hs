{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Networks as arrows.
--
-- A 'Net' is a stochastic function between (typed) tensors, with real
-- 'Category' and 'Arrow' instances — so networks compose with '>>>', run
-- branches in parallel with '***', fan out with '&&&', and @proc@ notation
-- works.  A ResNet skip connection is the arrow idiom it always was:
--
-- > residual f = (id &&& f) >>> arr (uncurry (+))
--
-- Design notes, learned from the earlier @feature/arrow@ attempt:
--
-- * Composition here is /value-level/: each piece of a chain carries its
--   input and output types, so the type of the middle of @f >>> g@ is
--   determined by the values, not searched for by instance resolution.  The
--   earlier attempt composed at the type level through 'HasForward', which
--   has no functional dependencies — every composition point became an
--   ambiguous type variable.  No @PartialTypeSignatures@ are needed here.
--
-- * Parameters stay in ordinary records with
--   @deriving (Generic, Parameterized)@ and are turned into wiring by a
--   plain function (@cnn :: CNN -> Net input output@).  Training with
--   'Torch.Typed.Optim.runStep' is unchanged.  The alternative — storing
--   parameters inside the arrow — needs either existentials (losing
--   'Torch.Typed.Parameter.Parameterized') or a type-level parameter list
--   (a graded category; possible, but not needed for the ergonomics).
--
-- * Ordinary typed operations lift with 'arr'
--   (@arr (conv2dForward \@'(1,1) \@'(1,1) conv)@ — output shapes are
--   computed by type families, so chains infer left to right), and whole
--   'Torch.Typed.NN.HasForward' modules lift with 'layer'.
module Torch.Typed.NN.Arrow
  ( Net (..),
    layer,
    residual,
    residualWith,
  )
where

import Control.Arrow
import Control.Category
import Torch.Typed.NN (HasForward (forwardStoch))
import Prelude hiding (id, (.))

-- | A network from @x@ to @y@: a stochastic function.  Pure operations lift
-- with 'arr'; modules with dropout-like behaviour keep their randomness in
-- 'IO' through 'layer'.
newtype Net x y = Net {runNet :: x -> IO y}

instance Category Net where
  id = Net pure
  Net g . Net f = Net (\x -> f x >>= g)

instance Arrow Net where
  arr f = Net (pure . f)
  first (Net f) = Net (\(x, y) -> (\x' -> (x', y)) <$> f x)
  second (Net f) = Net (\(x, y) -> (\y' -> (x, y')) <$> f y)

instance ArrowChoice Net where
  left (Net f) = Net (either (fmap Left . f) (pure . Right))

-- | Lift a module through its stochastic forward pass.
layer :: HasForward m x y => m -> Net x y
layer m = Net (forwardStoch m)

-- | A skip connection: @residual f@ computes @f x + x@.
residual :: Num a => Net a a -> Net a a
residual = residualWith id

-- | A skip connection with a projection on the skip path, as in ResNet
-- downsampling blocks: @residualWith skip f@ computes @f x + skip x@.
residualWith :: Num c => Net a c -> Net a c -> Net a c
residualWith skip f = (skip &&& f) >>> arr (uncurry (+))
