-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Pairwise
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
module Torch.Indef.Static.Tensor.Math.Pairwise where

import Numeric.Dimensions
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Pairwise as Dynamic

class Num real => Pairwise tensor real where
  -- | infix version of 'add'
  (^+) :: tensor -> real -> tensor
  -- | infix version of 'sub'
  (^-) :: tensor -> real -> tensor
  -- | infix version of 'mul'
  (^*) :: tensor -> real -> tensor
  -- | infix version of 'div'
  (^/) :: tensor -> real -> tensor

instance (Pairwise t0 r, Pairwise t1 r) => Pairwise (t0, t1) r where
  (a,b) ^+ v = (a ^+ v, b ^+ v)
  (a,b) ^- v = (a ^- v, b ^- v)
  (a,b) ^* v = (a ^* v, b ^* v)
  (a,b) ^/ v = (a ^/ v, b ^/ v)

infixl 7 ^*,^/
infixl 6 ^+,^-

instance Dimensions d => Pairwise (Tensor d) HsReal where
  (^+) = add
  (^-) = sub
  (^*) = mul
  (^/) = Torch.Indef.Static.Tensor.Math.Pairwise.div

-- | flipped version of '(^+)'
(+^) :: Pairwise ten real => real -> ten -> ten
(+^) = flip (^+)

-- | flipped version of '(^*)'
(*^) :: Pairwise ten real => real -> ten -> ten
(*^) = flip (^*)

-- | flipped version of '(^/)'
(/^) :: Pairwise ten real => real -> ten -> ten
(/^) = flip (^/)

-- | flipped version of '(^-)'
(-^) :: forall ten real . Pairwise ten real => real -> ten -> ten
(-^) v t = v +^ (negate (1::real) *^ t)

-- | static version of 'Dynamic.add_'
add_ :: Dimensions d => Tensor d -> HsReal -> IO ()
add_ t = Dynamic.add_ (asDynamic t)

-- | static version of 'Dynamic.add'
add :: Dimensions d => Tensor d -> HsReal -> Tensor d
add t v = asStatic $ Dynamic.add (asDynamic t) v

-- | static version of 'Dynamic.sub_'
sub_ :: Dimensions d => Tensor d -> HsReal -> IO ()
sub_ t = Dynamic.sub_ (asDynamic t)

-- | static version of 'Dynamic.sub'
sub :: Dimensions d => Tensor d -> HsReal -> Tensor d
sub t v = asStatic $ Dynamic.sub (asDynamic t) v

-- | static version of 'Dynamic.add_'
add_scaled_ :: Dimensions d => Tensor d -> HsReal -> HsReal -> IO ()
add_scaled_ t = Dynamic.add_scaled_ (asDynamic t)

-- | static version of 'Dynamic.sub_'
sub_scaled_ :: Dimensions d => Tensor d -> HsReal -> HsReal -> IO ()
sub_scaled_ t = Dynamic.sub_scaled_ (asDynamic t)

-- | static version of 'Dynamic.mul_'
mul_ :: Dimensions d => Tensor d -> HsReal -> IO ()
mul_ t = Dynamic.mul_ (asDynamic t)

-- | static version of 'Dynamic.mul'
mul :: Dimensions d => Tensor d -> HsReal -> Tensor d
mul t v = asStatic $ Dynamic.mul (asDynamic t) v

-- | static version of 'Dynamic.div_'
div_ t = Dynamic.div_ (asDynamic t)

-- | static version of 'Dynamic.div'
div t v = asStatic $ Dynamic.div (asDynamic t) v

-- | static version of 'Dynamic.lshift_'
lshift_ :: Dimensions d => Tensor d -> HsReal -> IO ()
lshift_ t = Dynamic.lshift_ (asDynamic t)

-- | static version of 'Dynamic.rshift_'
rshift_ :: Dimensions d => Tensor d -> HsReal -> IO ()
rshift_ t = Dynamic.rshift_ (asDynamic t)

-- | static version of 'Dynamic.fmod_'
fmod_ :: Dimensions d => Tensor d -> HsReal -> IO ()
fmod_ t = Dynamic.fmod_ (asDynamic t)

-- | static version of 'Dynamic.remainder_'
remainder_ :: Dimensions d => Tensor d -> HsReal -> IO ()
remainder_ t = Dynamic.remainder_ (asDynamic t)

-- | static version of 'Dynamic.bitand_'
bitand_ :: Dimensions d => Tensor d -> HsReal -> IO ()
bitand_ t = Dynamic.bitand_ (asDynamic t)

-- | static version of 'Dynamic.bitor_'
bitor_ :: Dimensions d => Tensor d -> HsReal -> IO ()
bitor_ t = Dynamic.bitor_ (asDynamic t)

-- | static version of 'Dynamic.bitxor_'
bitxor_ :: Dimensions d => Tensor d -> HsReal -> IO ()
bitxor_ t = Dynamic.bitxor_ (asDynamic t)


