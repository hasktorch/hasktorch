module Torch.Core.Tensor.Dynamic.DoubleMathSpec (spec) where

import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Dynamic.DoubleMath

import Torch.Prelude.Extras

main :: IO ()
main = hspec spec

spec :: Spec
spec =
  describe "scenario" $
    it "runs this scenario as expected without crashing" testScenario

testScenario :: Property
testScenario = monadicIO $ do

  -- check exception case
  let (m, v) = (td_init (D2 (3, 2)) 3.0 , td_init (D1 2) 2.0)
  run $ disp m
  run $ disp v
  run $ disp (td_mv m v)

  let (m, v) = (td_init (D3 (1, 3, 2)) 3.0 , td_init (D1 2) 2.0)

  run $ disp $ td_addr
    0.0 (td_init (D2 (3,2)) 0.0)
    1.0 (td_init (D1 3) 2.0) (td_init (D1 2) 3.0)

  run $ disp $ td_outer (td_init (D1 3) 2.0) (td_init (D1 2) 3.0)

