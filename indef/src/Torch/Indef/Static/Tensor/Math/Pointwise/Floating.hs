-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Pointwise.Floating
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.Pointwise.Floating where

import Numeric.Dimensions
import GHC.Int
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating as Dynamic

import Torch.Indef.Types

-- | Static version of 'Dynamic.cinv'
cinv :: Dimensions d => Tensor d -> Tensor d
cinv t = asStatic $ Dynamic.cinv (asDynamic t)

-- | Static version of 'Dynamic.sigmoid'
sigmoid :: Dimensions d => Tensor d -> Tensor d
sigmoid  t = asStatic $ Dynamic.sigmoid (asDynamic t)

-- | Static version of 'Dynamic.log'
log :: Dimensions d => Tensor d -> Tensor d
log  t = asStatic $ Dynamic.log (asDynamic t)

-- | Static version of 'Dynamic.lgamma'
lgamma :: Dimensions d => Tensor d -> Tensor d
lgamma  t = asStatic $ Dynamic.lgamma (asDynamic t)

-- | Static version of 'Dynamic.log1p'
log1p :: Dimensions d => Tensor d -> Tensor d
log1p  t = asStatic $ Dynamic.log1p (asDynamic t)

-- | Static version of 'Dynamic.exp'
exp :: Dimensions d => Tensor d -> Tensor d
exp  t = asStatic $ Dynamic.exp (asDynamic t)

-- | Static version of 'Dynamic.cos'
cos :: Dimensions d => Tensor d -> Tensor d
cos  t = asStatic $ Dynamic.cos (asDynamic t)

-- | Static version of 'Dynamic.acos'
acos :: Dimensions d => Tensor d -> Tensor d
acos  t = asStatic $ Dynamic.acos (asDynamic t)

-- | Static version of 'Dynamic.cosh'
cosh :: Dimensions d => Tensor d -> Tensor d
cosh  t = asStatic $ Dynamic.cosh (asDynamic t)

-- | Static version of 'Dynamic.sin'
sin :: Dimensions d => Tensor d -> Tensor d
sin  t = asStatic $ Dynamic.sin (asDynamic t)

-- | Static version of 'Dynamic.asin'
asin :: Dimensions d => Tensor d -> Tensor d
asin  t = asStatic $ Dynamic.asin (asDynamic t)

-- | Static version of 'Dynamic.sinh'
sinh :: Dimensions d => Tensor d -> Tensor d
sinh  t = asStatic $ Dynamic.sinh (asDynamic t)

-- | Static version of 'Dynamic.tan'
tan :: Dimensions d => Tensor d -> Tensor d
tan  t = asStatic $ Dynamic.tan (asDynamic t)

-- | Static version of 'Dynamic.atan'
atan :: Dimensions d => Tensor d -> Tensor d
atan  t = asStatic $ Dynamic.atan (asDynamic t)

-- | Static version of 'Dynamic.tanh'
tanh :: Dimensions d => Tensor d -> Tensor d
tanh  t = asStatic $ Dynamic.tanh (asDynamic t)

-- | Static version of 'Dynamic.erf'
erf :: Dimensions d => Tensor d -> Tensor d
erf  t = asStatic $ Dynamic.erf (asDynamic t)

-- | Static version of 'Dynamic.erfinv'
erfinv :: Dimensions d => Tensor d -> Tensor d
erfinv  t = asStatic $ Dynamic.erfinv (asDynamic t)

-- | Static version of 'Dynamic.pow'
pow :: Dimensions d => Tensor d -> HsReal -> Tensor d
pow  t v = asStatic $ Dynamic.pow (asDynamic t) v

-- | Static version of 'Dynamic.tpow'
tpow :: Dimensions d => HsReal -> Tensor d -> Tensor d
tpow v t = asStatic $ Dynamic.tpow v (asDynamic t)

-- | Static version of 'Dynamic.sqrt'
sqrt :: Dimensions d => Tensor d -> Tensor d
sqrt  t = asStatic $ Dynamic.sqrt (asDynamic t)

-- | Static version of 'Dynamic.rsqrt'
rsqrt :: Dimensions d => Tensor d -> Tensor d
rsqrt  t = asStatic $ Dynamic.rsqrt (asDynamic t)

-- | Static version of 'Dynamic.ceil'
ceil :: Dimensions d => Tensor d -> Tensor d
ceil  t = asStatic $ Dynamic.ceil (asDynamic t)

-- | Static version of 'Dynamic.floor'
floor :: Dimensions d => Tensor d -> Tensor d
floor  t = asStatic $ Dynamic.floor (asDynamic t)

-- | Static version of 'Dynamic.round'
round :: Dimensions d => Tensor d -> Tensor d
round  t = asStatic $ Dynamic.round (asDynamic t)

-- | Static version of 'Dynamic.trunc'
trunc :: Dimensions d => Tensor d -> Tensor d
trunc  t = asStatic $ Dynamic.trunc (asDynamic t)

-- | Static version of 'Dynamic.frac'
frac :: Dimensions d => Tensor d -> Tensor d
frac  t = asStatic $ Dynamic.frac (asDynamic t)

-- | Static version of 'Dynamic.lerp'
lerp :: Dimensions d => Tensor d -> Tensor d -> HsReal -> Tensor d
lerp a b v = asStatic $ Dynamic.lerp (asDynamic a) (asDynamic b) v

-- | Static version of 'Dynamic.atan2'
atan2 :: Dimensions d => Tensor d -> Tensor d -> Tensor d
atan2 a b = asStatic $ Dynamic.atan2 (asDynamic a) (asDynamic b)

