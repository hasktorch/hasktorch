-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Math.Random.TH
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- FIXME: copy-paste, or switch documentation to preference Static modules.
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Math.Random.TH where

import Numeric.Dimensions

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.Math.Random.TH as Dynamic
import qualified Torch.Types.TH as TH

-- | Statically typed version of 'Dynamic._rand'.
_rand :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
_rand t = Dynamic._rand (asDynamic t)

-- | Statically typed version of 'Dynamic._randn'.
_randn :: Dimensions d => Tensor d -> Generator -> TH.LongStorage -> IO ()
_randn t = Dynamic._randn (asDynamic t)

-- | Statically typed version of 'Dynamic._randperm'.
_randperm :: Dimensions d => Tensor d -> Generator -> Integer -> IO ()
_randperm t = Dynamic._randperm (asDynamic t)


