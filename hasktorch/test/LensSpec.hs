{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}

module LensSpec(spec) where

import Test.Hspec

import Torch.Lens
import GHC.Generics
import Control.Exception.Safe (catch,throwIO)
import Language.C.Inline.Cpp.Exceptions (CppException(..))

data WTree a w
  = Leaf a
  | Fork (WTree a w) (WTree a w)
  | WithWeight (WTree a w) w
  deriving (Generic, Show, Eq)

myTree = WithWeight (Fork (Leaf (Just "hello")) (Leaf Nothing)) "world"

instance {-# OVERLAPS #-} HasTypes String String where
  types_ = id

spec :: Spec
spec = describe "lens" $ do
  it "over for list" $ do
    over (types @String) (++ "!") ["hello"] `shouldBe` ["hello!"]
  it "over for tree" $ do
    over (types @String) (++ "!") myTree `shouldBe` WithWeight (Fork (Leaf (Just "hello!")) (Leaf Nothing)) "world!"
     
