module Torch.Types.THC.Random
  ( Generator(..)
  ) where

import Foreign
import Torch.Types.THC


newtype Generator = Generator
  { rng :: ForeignPtr CTHCGenerator
  } deriving (Eq, Show)


