module Extras
  ( doesn'tCrash
  , doesn'tCrashM
  ) where

doesn'tCrash :: a -> Bool
doesn'tCrash = const True

doesn'tCrashM :: Monad m => a -> m Bool
doesn'tCrashM = const (pure True)

