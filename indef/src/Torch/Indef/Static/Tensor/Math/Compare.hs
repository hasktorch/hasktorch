-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Compare
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Static.Tensor.Math.Compare
  ( ltValue, ltValueT, ltValueT_
  , leValue, leValueT, leValueT_
  , gtValue, gtValueT, gtValueT_
  , geValue, geValueT, geValueT_
  , neValue, neValueT, neValueT_
  , eqValue, eqValueT, eqValueT_
  ) where

import Numeric.Dimensions

import Torch.Indef.Mask
import Torch.Indef.Static.Tensor
import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Compare as Dynamic

-- | return a byte tensor which contains boolean values indicating the relation between a tensor and a given scalar.
ltValue, leValue, gtValue, geValue, neValue, eqValue
  :: Dimensions d
  => Tensor d -> HsReal -> MaskTensor d
ltValue a v = byteAsStatic $ Dynamic.ltValue (asDynamic a) v
leValue a v = byteAsStatic $ Dynamic.leValue (asDynamic a) v
gtValue a v = byteAsStatic $ Dynamic.gtValue (asDynamic a) v
geValue a v = byteAsStatic $ Dynamic.geValue (asDynamic a) v
neValue a v = byteAsStatic $ Dynamic.neValue (asDynamic a) v
eqValue a v = byteAsStatic $ Dynamic.eqValue (asDynamic a) v

-- | return a tensor which contains numeric values indicating the relation between a tensor and a given scalar.
-- 0 stands for false, 1 stands for true.
ltValueT, leValueT, gtValueT, geValueT, neValueT, eqValueT
  :: (Dimensions d)
  => Tensor d -> HsReal -> IO (Tensor d)
ltValueT a b = asStatic <$> Dynamic.ltValueT (asDynamic a) b
leValueT a b = asStatic <$> Dynamic.leValueT (asDynamic a) b
gtValueT a b = asStatic <$> Dynamic.gtValueT (asDynamic a) b
geValueT a b = asStatic <$> Dynamic.geValueT (asDynamic a) b
neValueT a b = asStatic <$> Dynamic.neValueT (asDynamic a) b
eqValueT a b = asStatic <$> Dynamic.eqValueT (asDynamic a) b

-- | mutate a tensor in-place with its numeric relation to a given scalar, where 0 stands for false and
-- 1 stands for true.
ltValueT_, leValueT_, gtValueT_, geValueT_, neValueT_, eqValueT_
  :: (Dimensions d)
  => Tensor d -> HsReal -> IO ()
ltValueT_ a b = Dynamic.ltValueT_ (asDynamic a) b
leValueT_ a b = Dynamic.leValueT_ (asDynamic a) b
gtValueT_ a b = Dynamic.gtValueT_ (asDynamic a) b
geValueT_ a b = Dynamic.geValueT_ (asDynamic a) b
neValueT_ a b = Dynamic.neValueT_ (asDynamic a) b
eqValueT_ a b = Dynamic.eqValueT_ (asDynamic a) b

