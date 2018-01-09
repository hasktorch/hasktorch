{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.Static.DoubleSpec (spec) where

import Control.Monad (replicateM, void)
import Foreign (Ptr)

import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dynamic.Double (printTensor)
import Torch.Core.Tensor.Static.Double
import qualified THRandom as R (c_THGenerator_new)

import Torch.Prelude.Extras
import Torch.Core.Random

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Eq instance" eqSpec
  describe "transpose"   transposeSpec
  describe "resize"      resizeSpec
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario
  describe "expand" $
    it "can expand a vector without crashing" expandScenario

testScenario :: Property
testScenario = monadicIO $ do
  run $ putStrLn "\n1 - new [2, 2]"
  run $ tds_p (tds_new :: TDS '[2, 2])
  run $ putStrLn "\n2 - new [2, 4]"
  run $ tds_p (tds_new :: TDS '[2, 4])
  run $ putStrLn "\n3 - new [2, 2, 2]"
  run $ tds_p (tds_new :: TDS '[2, 2, 2])
  run $ putStrLn "\n4 - new [8, 4]"
  run $ tds_p (tds_new :: TDS '[8, 4])
  run $ putStrLn "\n5 - init [3, 4]"
  run $ printTensor $ tds_toDynamic (tds_init 2.0 :: TDS '[3, 4])
  run $ putStrLn "\n6 - newClone [2, 3]"
  run $ tds_p $ tds_newClone (tds_init 2.0 :: TDS '[2, 3])

eqSpec :: Spec
eqSpec = do
  it "should be True"  $ (tds_init 4.0 :: TDS '[2,3]) == (tds_init 4.0 :: TDS '[2,3])
  it "should be False" $ (tds_init 3.0 :: TDS '[2,3]) /= (tds_init 1.0 :: TDS '[2,3])

transposeSpec :: Spec
transposeSpec = do
  runIO $ tds_p $ tds_trans . tds_trans . tds_trans $ (tds_init 3.0 :: TDS '[3,2])
  it "should transpose correctly" $ (tds_trans . tds_trans $ (tds_init 3.0 :: TDS '[3,2])) == (tds_init 3.0 :: TDS '[3,2])

resizeSpec :: Spec
resizeSpec = do
  runIO $ tds_p vec
  runIO $ tds_p mtx
 where
  vec = tds_fromList [1.0, 5.0, 2.0, 4.0, 3.0, 3.5] :: TDS '[6]
  mtx = tds_resize vec :: TDS '[3,2]

expandScenario :: IO ()
expandScenario = do
  let foo = tds_fromList [1,2,3,4] :: TDS '[4]
  let result = tds_expand foo :: TDS '[3, 4]
  tds_p result
