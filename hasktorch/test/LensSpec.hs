{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module LensSpec (spec) where

import Control.Exception.Safe (catch, throwIO)
import GHC.Generics
import Language.C.Inline.Cpp.Exceptions (CppException (..))
import Test.Hspec
import Torch.Lens

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
