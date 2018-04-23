-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Storage
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Re-exports of all CPU-based storage code
-------------------------------------------------------------------------------
module Torch.Storage ( module X ) where

import Torch.Types.TH as X

import Torch.Byte.Storage as X
import Torch.Char.Storage as X

import Torch.Short.Storage as X
import Torch.Int.Storage as X
import Torch.Long.Storage as X

import Torch.Float.Storage as X
import Torch.Double.Storage as X

