-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Pointwise
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.Tensor.Math.Pointwise where

import Numeric.Dimensions
import System.IO.Unsafe
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise as Dynamic

-- | Static version of 'Dynamic.sign'
sign :: Tensor d -> Tensor d
sign t = asStatic $ Dynamic.sign (asDynamic t)

-- | Static version of 'Dynamic.clamp'
clamp :: Tensor d -> HsReal -> HsReal -> Tensor d
clamp t a b = asStatic $ Dynamic.clamp (asDynamic t) a b

-- | Multiply elements of tensor2 by the scalar value and add it to tensor1.
-- The number of elements must match, but sizes do not matter.
--
-- Static version of 'Dynamic.cadd'.
cadd
  :: Tensor d  -- ^ tensor1
  -> HsReal    -- ^ scale term to multiply againts tensor2
  -> Tensor d  -- ^ tensor2
  -> Tensor d
cadd t v b = asStatic $ Dynamic.cadd (asDynamic t) v (asDynamic b)

-- | infix version of 'cadd' on dimension 1
(^+^) a b = cadd a 1 b

-- | Static version of 'Dynamic.csub'
csub
  :: Tensor d  -- ^ tensor1
  -> HsReal    -- ^ scale term to multiply againts tensor2
  -> Tensor d  -- ^ tensor2
  -> Tensor d
csub t v b = asStatic $ Dynamic.csub (asDynamic t) v (asDynamic b)

-- | infix version of 'csub' on dimension 1
(^-^) a b = csub a 1 b

-- | Static version of 'Dynamic.cmul'
cmul :: Tensor d -> Tensor d -> Tensor d
cmul t1 t2 = asStatic $ Dynamic.cmul (asDynamic t1) (asDynamic t2)

-- | square a tensor
square t = cmul t t

-- | infix version of 'cmul'.
(^*^) a b = cmul a b

-- | Static version of 'Dynamic.cdiv'
cdiv :: Tensor d -> Tensor d -> Tensor d
cdiv t1 t2 = asStatic $ Dynamic.cdiv (asDynamic t1) (asDynamic t2)

-- | Infix version of 'cdiv'.
(^/^) a b = cdiv a b

-- | Static version of 'Dynamic.cpow'
cpow :: Tensor d -> Tensor d -> Tensor d
cpow t1 t2 = asStatic $ Dynamic.cpow (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.clshift'
clshift :: Tensor d -> Tensor d -> Tensor d
clshift t1 t2 = asStatic $ Dynamic.clshift (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.crshift'
crshift :: Tensor d -> Tensor d -> Tensor d
crshift t1 t2 = asStatic $ Dynamic.crshift (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.cfmod'
cfmod :: Tensor d -> Tensor d -> Tensor d
cfmod t1 t2 = asStatic $ Dynamic.cfmod (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.cremainder'
cremainder :: Tensor d -> Tensor d -> Tensor d
cremainder t1 t2 = asStatic $ Dynamic.cremainder (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.cmax'
cmax :: Tensor d -> Tensor d -> Tensor d
cmax  a b = asStatic $ Dynamic.cmax (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cmin'
cmin :: Tensor d -> Tensor d -> Tensor d
cmin  a b = asStatic $ Dynamic.cmin (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cbitand'
cbitand :: Tensor d -> Tensor d -> Tensor d
cbitand  a b = asStatic $ Dynamic.cbitand (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cbitor'
cbitor :: Tensor d -> Tensor d -> Tensor d
cbitor  a b = asStatic $ Dynamic.cbitor (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cbitxor'
cbitxor :: Tensor d -> Tensor d -> Tensor d
cbitxor  a b = asStatic $ Dynamic.cbitxor (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.addcmul'
addcmul  a v b c = asStatic $ Dynamic.addcmul (asDynamic a) v (asDynamic b) (asDynamic c)

-- | Static version of 'Dynamic.addcdiv'
addcdiv  a v b c = asStatic $ Dynamic.addcdiv (asDynamic a) v (asDynamic b) (asDynamic c)




