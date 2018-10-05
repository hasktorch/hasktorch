{- OPTIONS_GHC -F -pgmF hspec-discover #-}
module Main where

import Test.Hspec
import qualified Torch.Indef.StorageSpec as Storage

main :: IO ()
main = hspec $ do
  describe "Storage" Storage.spec


