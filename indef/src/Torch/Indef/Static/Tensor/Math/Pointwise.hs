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
sign :: (Dimensions d) => Tensor d -> IO (Tensor d)
sign t = asStatic <$> Dynamic.sign (asDynamic t)

-- | Static version of 'Dynamic.clamp'
clamp :: (Dimensions d) => Tensor d -> HsReal -> HsReal -> IO (Tensor d)
clamp t a b = asStatic <$> Dynamic.clamp (asDynamic t) a b

-- | Static version of 'Dynamic.cadd'
cadd :: (Dimensions d) => Tensor d -> HsReal -> Tensor d -> IO (Tensor d)
cadd t v b = asStatic <$> Dynamic.cadd (asDynamic t) v (asDynamic b)

-- | infix version of 'cadd' on dimension 1
(^+^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^+^) a b = unsafePerformIO $ cadd a 1 b
{-# NOINLINE (^+^) #-}

-- | Static version of 'Dynamic.csub'
csub :: (Dimensions d) => Tensor d -> HsReal -> Tensor d -> IO (Tensor d)
csub t v b = asStatic <$> Dynamic.csub (asDynamic t) v (asDynamic b)

-- | infix version of 'csub' on dimension 1
(^-^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^-^) a b = unsafePerformIO $ csub a 1 b
{-# NOINLINE (^-^) #-}

-- | Static version of 'Dynamic.cmul'
cmul :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cmul t1 t2 = asStatic <$> Dynamic.cmul (asDynamic t1) (asDynamic t2)

-- | square a tensor
square :: (Dimensions d) => Tensor d -> IO (Tensor d)
square t = cmul t t

-- | infix version of 'cmul'.
(^*^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^*^) a b = unsafePerformIO $ cmul a b
{-# NOINLINE (^*^) #-}

-- | Static version of 'Dynamic.cdiv'
cdiv :: Dimensions d => Tensor d -> Tensor d -> IO (Tensor d)
cdiv  t1 t2 = asStatic <$> Dynamic.cdiv (asDynamic t1) (asDynamic t2)

-- | Infix version of 'cdiv'.
(^/^) :: (Dimensions d) => Tensor d -> Tensor d -> Tensor d
(^/^) a b = unsafePerformIO $ cdiv a b
{-# NOINLINE (^/^) #-}

-- | Static version of 'Dynamic.cpow'
cpow :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cpow t1 t2 = asStatic <$> Dynamic.cpow (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.clshift'
clshift :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
clshift t1 t2 = asStatic <$> Dynamic.clshift (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.crshift'
crshift :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
crshift t1 t2 = asStatic <$> Dynamic.crshift (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.cfmod'
cfmod :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cfmod t1 t2 = asStatic <$> Dynamic.cfmod (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.cremainder'
cremainder :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cremainder t1 t2 = asStatic <$> Dynamic.cremainder (asDynamic t1) (asDynamic t2)

-- | Static version of 'Dynamic.cmax'
cmax :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cmax  a b = asStatic <$> Dynamic.cmax (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cmin'
cmin :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cmin  a b = asStatic <$> Dynamic.cmin (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cbitand'
cbitand :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cbitand  a b = asStatic <$> Dynamic.cbitand (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cbitor'
cbitor :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cbitor  a b = asStatic <$> Dynamic.cbitor (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.cbitxor'
cbitxor :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
cbitxor  a b = asStatic <$> Dynamic.cbitxor (asDynamic a) (asDynamic b)

-- | Static version of 'Dynamic.addcmul'
addcmul :: (Dimensions d) => Tensor d -> HsReal -> Tensor d -> Tensor d -> IO (Tensor d)
addcmul  a v b c = asStatic <$> Dynamic.addcmul (asDynamic a) v (asDynamic b) (asDynamic c)

-- | Static version of 'Dynamic.addcdiv'
addcdiv :: (Dimensions d) => Tensor d -> HsReal -> Tensor d -> Tensor d -> IO (Tensor d)
addcdiv  a v b c = asStatic <$> Dynamic.addcdiv (asDynamic a) v (asDynamic b) (asDynamic c)




