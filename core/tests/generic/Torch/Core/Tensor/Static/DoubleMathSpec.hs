{-# LANGUAGE DataKinds #-}
module Torch.Core.Tensor.Static.DoubleMathSpec where

import Torch.Core.Tensor.Static.Double (TDS, tds_init, tds_p)
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Random (newRNG)

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario


testScenario :: Property
testScenario = monadicIO $ do
  run $ putStrLn "initialization"
  let t = (tds_init 3.0 :: TDS '[3,2])
  run $ tds_p t
  run $ putStrLn "trace "
  run $ print $ tds_trace t
  run $ putStrLn "num el"
  run $ print $ tds_numel t
  run $ putStrLn "Dot product"
  run $ print ((tds_init 2.0 :: TDS '[5]) <.> (tds_init 2.0 :: TDS '[5]))

  -- .33..
  run $ tds_p $ tds_cinv (tds_init 3.0 :: TDS '[5])

  -- 1.25
  run $ tds_p $ 5.0 /^ (tds_init 4.0 :: TDS '[10])

  run $ tds_p (tds_outer (tds_init 2.0) (tds_init 3.0) :: TDS '[3,2])

  run . putStrLn . show $ tds_equal (tds_init 3.0 :: TDS '[4]) (tds_init 2.0 :: TDS '[4])
  run . putStrLn . show $ tds_equal (tds_init 3.0 :: TDS '[4]) (tds_init 3.0 :: TDS '[4])
