{-# LANGUAGE DataKinds #-}
module Torch.Core.Tensor.Static.DoubleLapackSpec where

import Torch.Core.Tensor.Static.Double (TDS, StaticTensor(..), tds_fromList)
import Torch.Core.Tensor.Static.DoubleMath ((!*!))
import Torch.Core.Tensor.Static.DoubleLapack

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario

testScenario :: Property
testScenario = monadicIO $ do
  let a = (tds_init 2.0 :: TDS '[2, 2]) !*! (tds_init 2.0 :: TDS '[2, 2])
  let b = tds_new :: TDS '[2]
  run $ tds_p a
  run $ tds_p b

  let x = (tds_fromList [10,5,7,4] :: TDS [2,2])
  let c = tds_init 1.0 :: TDS '[2]
  let (resb, resa) = tds_gesv c x
  run $ tds_p resb
  run $ tds_p resa

  -- let potrf = tds_potrf x Upper

  -- crashes
  let (resb, resa) = tds_gesv b a
  run $ tds_p resb
  run $ tds_p resa

