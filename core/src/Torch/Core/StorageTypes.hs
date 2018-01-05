{-# LANGUAGE TypeFamilies #-}

module Torch.Core.StorageTypes (
  StorageSize(..),
  StorageDouble(..),
  StorageLong(..)
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import THTypes
import THDoubleStorage
import Torch.Core.Tensor.Dynamic.Generic.Internal (THTensor(..), THPtrType)

-- TODO - consider a shared backend type for TensorDim and StorageSize
data StorageSize a =
  S0
  | S1 { s1 :: a }
  | S2 { s2 :: (a, a) }
  | S3 { s3 :: (a, a, a) }
  | S4 { s4 :: (a, a, a, a) }
  deriving (Eq, Show)

instance Functor StorageSize where
  fmap f S0 = S0
  fmap f (S1 s1) = S1 (f s1)
  fmap f (S2 (s1, s2)) = S2 ((f s1), (f s2))
  fmap f (S3 (s1, s2, s3)) = S3 ((f s1), (f s2), (f s3))
  fmap f (S4 (s1, s2, s3, s4)) = S4 ((f s1), (f s2), (f s3), (f s4))

instance Foldable StorageSize where
  foldr func val (S0) = val
  foldr func val (S1 s1) = foldr func val [s1]
  foldr func val (S2 (s1, s2)) = foldr func val [s1, s2]
  foldr func val (S3 (s1, s2, s3)) = foldr func val [s1, s2, s3]
  foldr func val (S4 (s1, s2, s3, s4)) = foldr func val [s1, s2, s3, s4]

data StorageDouble = StorageDouble
  { sdStorage :: !(ForeignPtr CTHDoubleStorage)
  -- , sdSize :: !(StorageSize Double)
  } deriving (Eq, Show)

data StorageLong = StorageLong
  { slStorage :: !(ForeignPtr CTHLongStorage)
  -- , slSize :: !(StorageSize Int)
  } deriving (Eq, Show)

instance THTensor StorageDouble where
  construct = StorageDouble
  getForeign = sdStorage
type instance THPtrType StorageDouble = CTHDoubleStorage

instance THTensor StorageLong where
  construct = StorageLong
  getForeign = slStorage
type instance THPtrType StorageLong = CTHLongStorage
