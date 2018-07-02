{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Static.NN.LinearSpec where

import Test.Hspec
import Torch.Double
import Numeric.Backprop

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  let Just (x :: DoubleTensor '[2, 4]) = fromList [-4..4-1]
  describe "a single linear layer" $ do
    it "runs the absolute function" $ do
      y <- tensordata =<< abs_updateOutput x
      y `shouldSatisfy` all (>= 0)

-- linear
--   :: forall s i o
--   .  Reifies s W
--   => All KnownDim '[i,o]
--   => BVar s (Linear i o)
--   -> BVar s (Tensor '[i])
--   -> BVar s (Tensor '[o])
-- linear = liftOp2 $ op2 $ \l i -> (transpose2d (weights l) `mv` i + bias l, go l i)
--   where
--     go :: Linear i o -> Tensor '[i] -> Tensor '[o] -> (Linear i o, Tensor '[i])
--     go (Linear (w, b)) i gout = (Linear (i `outer` b', b'), w `mv` b')
--       where
--         b' = gout - b


