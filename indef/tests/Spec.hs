{- OPTIONS_GHC -F -pgmF hspec-discover #-}
module Main where

import Test.Hspec
import qualified Torch.Indef.StorageSpec as Storage
import qualified Torch.Indef.Dynamic.TensorSpec as Dynamic.Tensor
import qualified Torch.Indef.Static.TensorSpec as Static.Tensor

main :: IO ()
main = hspec $ do
  describe "Storage"        Storage.spec
  describe "Dynamic.Tensor" Dynamic.Tensor.spec
  describe "Static.Tensor"  Static.Tensor.spec


