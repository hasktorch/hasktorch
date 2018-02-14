module THRandomTypes where

import Foreign
import THTypes

newtype Generator = Generator
  { rng :: ForeignPtr CTHGenerator
  } deriving (Eq, Show)


