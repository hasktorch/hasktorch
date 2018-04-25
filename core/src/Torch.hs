-------------------------------------------------------------------------------
-- |
-- Module    :  Torch
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Re-exports of all static CPU-based tensors
-------------------------------------------------------------------------------
module Torch ( module X ) where

import Torch.Dimensions as X
import Torch.Types.TH as X
import Torch.Storage as X

import Torch.Byte.Static as X
import Torch.Char.Static as X

import Torch.Short.Static as X
import Torch.Int.Static   as X
import Torch.Long.Static  as X

import Torch.Float.Static  as X
import Torch.Float.StaticRandom  as X
import Torch.Double.Static as X
import Torch.Double.StaticRandom as X

