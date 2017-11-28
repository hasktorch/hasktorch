{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Dynamic.DoubleSpec (spec) where

import Control.Monad (replicateM, void)
import Foreign (Ptr)
import Test.Hspec
import Test.QuickCheck
import Test.QuickCheck.Monadic
import Debug.Trace

import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dynamic.Double
import qualified THRandom as R (c_THGenerator_new)

import Torch.Core.Random
import Extras
import Orphans ()

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario

testScenario :: Property
testScenario = monadicIO $ do
  let foo = td_new (D1 5)
  let t = td_init (D2 (5, 2)) 3.0
  run (td_p (td_transpose 1 0 (td_transpose 1 0 t)))
