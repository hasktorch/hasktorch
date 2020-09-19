{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module NNSpec(spec) where

import Test.Hspec
import Control.Exception.Safe
import Control.Monad.State.Strict

import Torch.Tensor
import Torch.NN
import GHC.Generics

spec :: Spec
spec = do
  it "create flatten-parameters of Linear" $ do
    init <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    init2 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    length (flattenParameters init) `shouldBe` 2
    length (flattenParameters (fst (flip runState (flattenParameters init2) (_replaceParameters init)))) `shouldBe` 2
  it "create flatten-parameters of [Linear]" $ do
    i0 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    i1 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    i2 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    i3 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    let init = [i0,i1]
        init2 = [i2,i3]
    length (flattenParameters init) `shouldBe` 4
    length (flattenParameters (fst (flip runState (flattenParameters init2) (_replaceParameters init)))) `shouldBe` 4
