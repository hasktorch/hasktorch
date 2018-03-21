{-# OPTIONS_GHC -Wno-orphans #-}
module CodeGen.Instances where

import Test.QuickCheck

import CodeGen.Types

instance Arbitrary LibType where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary RawTenType where
  arbitrary = arbitraryBoundedEnum

instance Arbitrary TenType where
  arbitrary = fmap Pair $ (,)
    <$> arbitrary
    <*> arbitrary
