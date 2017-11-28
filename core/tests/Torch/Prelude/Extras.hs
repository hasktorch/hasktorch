module Torch.Prelude.Extras
  ( module X
  , doesn'tCrash
  , doesn'tCrashM
  ) where

import Test.Hspec as X
import Test.QuickCheck as X
import Test.QuickCheck.Monadic as X

import Orphans as X

doesn'tCrash :: a -> Bool
doesn'tCrash = const True

doesn'tCrashM :: Monad m => a -> m Bool
doesn'tCrashM = const (pure True)

