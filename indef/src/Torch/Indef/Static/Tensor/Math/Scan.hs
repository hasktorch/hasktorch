-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Scan
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.Scan where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Scan as Dynamic

-- | Static call to 'Dynamic._cumsum'
_cumsum :: Tensor d -> Tensor d -> DimVal -> IO ()
_cumsum r t = Dynamic._cumsum (asDynamic r) (asDynamic t)

-- | Static call to 'Dynamic._cumprod'
_cumprod :: Tensor d -> Tensor d -> DimVal -> IO ()
_cumprod r t = Dynamic._cumprod (asDynamic r) (asDynamic t)

