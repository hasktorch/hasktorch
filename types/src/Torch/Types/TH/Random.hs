module Torch.Types.TH.Random
  ( Generator(..)
  ) where

import Foreign
import Torch.Types.TH


newtype Generator = Generator
  { rng :: ForeignPtr CTHGenerator
  } deriving (Eq, Show)


