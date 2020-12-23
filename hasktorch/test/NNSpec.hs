{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeSynonymInstances #-}

module NNSpec(spec) where

import Test.Hspec
import Control.Exception.Safe
import Control.Monad.State.Strict

import Torch.Tensor
import Torch.TensorFactories
import Torch.Traversable
import Torch.Autograd
import Torch.NN
import GHC.Generics

data CustomlinearSpec = CustomlinearSpec
  { in_features :: Int,
    out_features :: Int
  }
  deriving (Show, Eq)

data Customlinear = Customlinear
  { customWeight :: Parameter,
    customBias :: Parameter
  }
  deriving (Show, Generic)

instance GTraversable Parameter Customlinear where
  gflatten Customlinear {..} = [customWeight, customBias, customBias] -- These duplicated biases are for testing!!
  gupdate _ = do
    customWeight <- gpop
    customBias <- gpop
    _ <- gpop
    return $ Customlinear {..}

ctol layer = Linear (customWeight layer) (customBias layer)

ltoc layer = Customlinear (weight layer) (bias layer)

ltoc2 layer = Customlinear2 (weight layer) (bias layer)

instance Randomizable CustomlinearSpec Customlinear where
  sample (CustomlinearSpec a b) = do
    v <- sample (LinearSpec a b)
    return $ ltoc v

instance Randomizable Customlinear2Spec Customlinear2 where
  sample (Customlinear2Spec a b) = do
    v <- sample (LinearSpec a b)
    return $ ltoc2 v

customlinear :: Customlinear -> Tensor -> Tensor
customlinear layer input = linear (ctol layer) input

data Customlinear2Spec = Customlinear2Spec
  { in_features2 :: Int,
    out_features2 :: Int
  }
  deriving (Show, Eq)

data Customlinear2 = Customlinear2
  { customWeight2 :: Parameter,
    customBias2 :: Parameter
  }
  deriving (Show)  -- Not deriving Generic is for for testing!!

instance GTraversable Parameter Customlinear2 where
  gflatten Customlinear2 {..} = [customWeight2, customBias2, customBias2, customBias2]
  gupdate _ = do
    customWeight2 <- gpop
    customBias2 <- gpop
    _ <- gpop
    _ <- gpop
    return $ Customlinear2 {..}


spec :: Spec
spec = do
  it "create flatten-parameters of Linear" $ do
    init <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    init2 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    length (flattenParameters init) `shouldBe` 2
    length (flattenParameters (fst (flip runState (flattenParameters init2) (gupdate init)))) `shouldBe` 2
  it "create flatten-parameters of [Linear]" $ do
    i0 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    i1 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    i2 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    i3 <- sample $ LinearSpec { in_features = 3, out_features = 1 }
    let init = [i0,i1]
        init2 = [i2,i3]
    length (flattenParameters init) `shouldBe` 4
    length (flattenParameters (fst (flip runState (flattenParameters init2) (gupdate init)))) `shouldBe` 4
  it "create flatten-parameters of (Parameter,Parameter)" $ do
    i0 <- makeIndependent $ zeros' [2,2]
    i1 <- makeIndependent $ zeros' [2,2]
    i2 <- makeIndependent $ zeros' [2,2]
    let init = (i0,i1)
        init2 = (i0,i1,i2)
    length (flattenParameters init) `shouldBe` 2
    length (flattenParameters init2) `shouldBe` 3

  it "create flatten-parameters of Customlinear" $ do
    init <- sample $ CustomlinearSpec { in_features = 3, out_features = 1 }
    length (flattenParameters init) `shouldBe` 3
    shape (customlinear init (ones' [2,3])) `shouldBe` [2,1]
  it "create flatten-parameters of [Customlinear]" $ do
    i0 <- sample $ CustomlinearSpec 3 1
    i1 <- sample $ CustomlinearSpec 3 1 
    i2 <- sample $ CustomlinearSpec 3 1
    i3 <- sample $ CustomlinearSpec 3 1
    let init = [i0,i1]
        init2 = [i2,i3]
    length (flattenParameters init) `shouldBe` 6
    length (flattenParameters (fst (flip runState (flattenParameters init2) (gupdate init)))) `shouldBe` 6

  it "create flatten-parameters of Customlinear2" $ do
    init <- sample $ Customlinear2Spec 3 1
    length (flattenParameters init) `shouldBe` 4
