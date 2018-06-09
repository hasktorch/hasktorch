-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Masked
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Masked where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Masked as Dynamic

-- | Static call to 'Dynamic._maskedFill'
_maskedFill t m v = Dynamic._maskedFill (asDynamic t) (byteAsDynamic m) v
-- | Static call to 'Dynamic._maskedCopy'
_maskedCopy r m t = Dynamic._maskedCopy (asDynamic r) (byteAsDynamic m) (asDynamic t)
-- | Static call to 'Dynamic._maskedSelect'
_maskedSelect r s m = Dynamic._maskedSelect (asDynamic r) (asDynamic s) (byteAsDynamic m)


