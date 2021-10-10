{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module NNSpec (spec) where

import Control.Exception.Safe
import Control.Monad.State.Strict
import GHC.Generics
import Test.Hspec
import Torch.Autograd
import Torch.NN
import Torch.Tensor
import Torch.TensorFactories

spec :: Spec
spec = do
  it "create flatten-parameters of Linear" $ do
    init <- sample $ LinearSpec {in_features = 3, out_features = 1}
    init2 <- sample $ LinearSpec {in_features = 3, out_features = 1}
    length (flattenParameters init) `shouldBe` 2
    length (flattenParameters (fst (flip runState (flattenParameters init2) (_replaceParameters init)))) `shouldBe` 2
  it "create flatten-parameters of [Linear]" $ do
    i0 <- sample $ LinearSpec {in_features = 3, out_features = 1}
    i1 <- sample $ LinearSpec {in_features = 3, out_features = 1}
    i2 <- sample $ LinearSpec {in_features = 3, out_features = 1}
    i3 <- sample $ LinearSpec {in_features = 3, out_features = 1}
    let init = [i0, i1]
        init2 = [i2, i3]
    length (flattenParameters init) `shouldBe` 4
    length (flattenParameters (fst (flip runState (flattenParameters init2) (_replaceParameters init)))) `shouldBe` 4
  it "create flatten-parameters of (Parameter,Parameter)" $ do
    i0 <- makeIndependent $ zeros' [2, 2]
    i1 <- makeIndependent $ zeros' [2, 2]
    i2 <- makeIndependent $ zeros' [2, 2]
    let init = (i0, i1)
        init2 = (i0, i1, i2)
    length (flattenParameters init) `shouldBe` 2
    length (flattenParameters init2) `shouldBe` 3
