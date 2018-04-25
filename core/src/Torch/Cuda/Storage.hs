-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Cuda.Storage
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Re-exports of all CUDA-based storage code
-------------------------------------------------------------------------------
module Torch.Cuda.Storage ( module X ) where

import Torch.Types.THC

import qualified Torch.Cuda.Byte.Storage as X
import qualified Torch.Cuda.Char.Storage as X

import qualified Torch.Cuda.Short.Storage as X
import qualified Torch.Cuda.Int.Storage as X
import qualified Torch.Cuda.Long.Storage as X

import qualified Torch.Cuda.Double.Storage as X
