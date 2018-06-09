-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Floating
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.Floating where

import Numeric.Dimensions
import GHC.Int

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Floating as Dynamic

-- | Static call to 'Dynamic.linspace'
linspace :: Dimensions d => HsReal -> HsReal -> Int64 -> IO (Tensor d)
linspace a b c = asStatic <$> Dynamic.linspace a b c

-- | Static call to 'Dynamic.linspace_'
linspace_ :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
linspace_ r = Dynamic.linspace_ (asDynamic r)

-- | Static call to 'Dynamic.logspace'
logspace :: Dimensions d => HsReal -> HsReal -> Int64 -> IO (Tensor d)
logspace a b c = asStatic <$> Dynamic.logspace a b c

-- | Static call to 'Dynamic.logspace_'
logspace_ :: Dimensions d => Tensor d -> HsReal -> HsReal -> Int64 -> IO ()
logspace_ r = Dynamic.logspace_ (asDynamic r)

