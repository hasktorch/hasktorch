{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Distributions.BernoulliSpec (spec) where

import GHC.Exts
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.Distributions.Bernoulli
import qualified Torch.Distributions.Constraints as Constraints
import Torch.Distributions.Distribution
import qualified Torch.Functional as F
import qualified Torch.Tensor as D
import Torch.Typed.Tensor

type Tnsr dtype shape = Tensor '( 'D.CPU, 0) dtype shape

spec :: Spec
spec = do
  let ps = [0.8, 0.2 :: Float]
  let p = D.asTensor ps
  let d = fromProbs p

  it "batchShape" $ do
    batchShape d `shouldBe` []

  it "eventShape" $ do
    eventShape d `shouldBe` []

  it "probs" $ do
    -- putStrLn . show $ probs d
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ probs d
    toList (Just t) `shouldBe` ps

  it "expand" $ do
    -- putStrLn . show $ expand d [2]
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ probs $ expand d [2]
    toList (Just t) `shouldBe` ps

  it "support" $ do
    -- putStrLn . show $ support d $ D.asTensor [0.0, 0.5, 1.0, 2.0 :: Float]
    let t :: Tnsr 'D.Bool '[4] =
          UnsafeMkTensor
            . support d
            $ D.asTensor [0.0, 0.5, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [True, False, True, False]

  it "mean" $ do
    -- putStrLn . show $ mean d
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ mean d
    toList (Just t) `shouldBe` ps

  it "variance" $ do
    -- putStrLn . show $ variance d
    F.allclose (variance d) (D.asTensor [0.16, 0.16 :: Float]) 0.01 0.01 False `shouldBe` True

  it "sample" $ do
    -- t <- sample d [2]
    -- putStrLn . show $ t
    t :: Tnsr 'D.Bool '[2] <- UnsafeMkTensor . Constraints.boolean <$> sample d [2]
    toList (Just t) `shouldBe` [True, True]

  it "logProb" $ do
    -- putStrLn . show $ logProb d $ D.asTensor [[0.3, 0.5 :: Float]]
    let t :: Tnsr 'D.Float '[1, 2] = UnsafeMkTensor $ logProb d $ D.asTensor [[0.3, 0.5 :: Float]]
    F.allclose (toDynamic t) (D.asTensor [[-0.6749387, -0.7530129 :: Float]]) 0.001 0.001 False `shouldBe` True

  it "entropy" $ do
    -- putStrLn . show $ entropy d
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ entropy d
    F.allclose (toDynamic t) (D.asTensor [0.7233937, 0.5433219 :: Float]) 0.0001 0.0001 False `shouldBe` True

  it "enumerateSupport" $ do
    -- putStrLn . show $ enumerateSupport d False
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ enumerateSupport d False
    toList (Just t) `shouldBe` [0.0, 1.0]
