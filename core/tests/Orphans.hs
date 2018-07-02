{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
module Orphans () where

import Data.Function (on)
import Test.QuickCheck

instance Num a => Num (Positive a) where
  (-)         = Positive .: ((-) `on` getPositive)
  (+)         = Positive .: ((+) `on` getPositive)
  (*)         = Positive .: ((*) `on` getPositive)
  abs         = Positive . abs . getPositive
  signum      = Positive . signum . getPositive
  fromInteger = Positive . fromInteger

(.:) = (.) . (.)


