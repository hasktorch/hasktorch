{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Shape-selective structure traversals: the generic 'types' traversal from
-- "Torch.Lens", instantiated at /typed/ tensor targets, visits exactly the
-- tensors of the named device, dtype and shape inside a structure.
module Torch.Typed.LensSpec (spec) where

import GHC.Generics
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.Lens (flattenValues, over, replaceValues, types)
import Torch.NN (sample)
import qualified Torch.Tensor as T
import Torch.Typed.Factories (ones, zeros)
import Torch.Typed.Lens ()
import Torch.Typed.NN (Linear, LinearSpec (..))
import Torch.Typed.Parameter (Parameter)
import Torch.Typed.Tensor

data Net = Net
  { l1 :: Tensor '(D.CPU, 0) 'D.Float '[2, 3],
    l2 :: Tensor '(D.CPU, 0) 'D.Float '[3, 4],
    l3 :: Tensor '(D.CPU, 0) 'D.Float '[2, 3],
    steps :: Int
  }
  deriving (Generic)

net :: Net
net = Net ones ones zeros 7

elems :: Tensor '(D.CPU, 0) 'D.Float sh -> [Float]
elems = T.asValue . T.reshape [-1] . toDynamic

spec :: Spec
spec = do
  describe "types at typed tensors" $ do
    it "selects tensors by their shape" $ do
      length (flattenValues (types @(Tensor '(D.CPU, 0) 'D.Float '[2, 3])) net) `shouldBe` 2
      length (flattenValues (types @(Tensor '(D.CPU, 0) 'D.Float '[3, 4])) net) `shouldBe` 1
      length (flattenValues (types @(Tensor '(D.CPU, 0) 'D.Float '[5, 5])) net) `shouldBe` 0
    it "rewrites only the tensors of the targeted shape" $ do
      let net' = over (types @(Tensor '(D.CPU, 0) 'D.Float '[2, 3])) (+ 1) net
      elems (l1 net') `shouldBe` replicate 6 2 -- was ones
      elems (l3 net') `shouldBe` replicate 6 1 -- was zeros
      elems (l2 net') `shouldBe` replicate 12 1 -- untouched
      steps net' `shouldBe` 7
    it "replaces specific structures wholesale" $ do
      let net' = replaceValues (types @(Tensor '(D.CPU, 0) 'D.Float '[2, 3])) net [zeros, ones]
      elems (l1 net') `shouldBe` replicate 6 0
      elems (l3 net') `shouldBe` replicate 6 1
      elems (l2 net') `shouldBe` replicate 12 1
  describe "types at typed parameters" $ do
    it "selects parameters of a model by their shape" $ do
      (model :: Linear 10 5 'D.Float '(D.CPU, 0)) <- sample LinearSpec
      length (flattenValues (types @(Parameter '(D.CPU, 0) 'D.Float '[5, 10])) model) `shouldBe` 1
      length (flattenValues (types @(Parameter '(D.CPU, 0) 'D.Float '[5])) model) `shouldBe` 1
      length (flattenValues (types @(Parameter '(D.CPU, 0) 'D.Float '[10, 5])) model) `shouldBe` 0
