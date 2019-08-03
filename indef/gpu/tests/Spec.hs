{- OPTIONS_GHC -F -pgmF hspec-discover #-}
module Main where

import Test.Hspec
import qualified Torch.Indef.StorageSpec             as Storage
import qualified Torch.Indef.Dynamic.TensorSpec      as Dynamic.Tensor
import qualified Torch.Indef.Dynamic.Tensor.MathSpec as Dynamic.Tensor.Math
import qualified Torch.Indef.Static.TensorSpec       as Static.Tensor
import qualified Torch.Indef.Static.Tensor.MathSpec  as Static.Tensor.Math

main :: IO ()
main = hspec $ do
  describe "Storage"             Storage.spec
  describe "Dynamic.Tensor"      Dynamic.Tensor.spec
  describe "Dynamic.Tensor.Math" Dynamic.Tensor.Math.spec
  describe "Static.Tensor"       Static.Tensor.spec
  describe "Static.Tensor.Math"  Static.Tensor.Math.spec
