-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Mode
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Mode where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Mode as Dynamic

-- | Static call to 'Dynamic._mode'
_mode :: (Tensor d, IndexTensor '[n]) -> Tensor d -> DimVal -> Maybe KeepDim -> IO ()
_mode (r, ix) t = Dynamic._mode (asDynamic r, longAsDynamic ix) (asDynamic t)

