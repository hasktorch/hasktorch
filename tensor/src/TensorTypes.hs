module TensorTypes (
  TensorDim(..),
  TensorDouble_(..)
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import THTypes
import THDoubleTensor

data TensorDim a =
  D0
  | D1 { d1_1 :: a }
  | D2 { d2_1 :: a, d2_2 :: a }
  | D3 { d3_1 :: a, d3_2 :: a, d3_3 :: a }
  | D4 { d4_1 :: a, d4_2 :: a, d4_3 :: a, d4_4 :: a }
  deriving (Eq, Show)

instance Functor TensorDim where
  fmap f D0 = D0
  fmap f (D1 d1) = D1 (f d1)
  fmap f (D2 d1 d2) = D2 (f d1) (f d2)
  fmap f (D3 d1 d2 d3) = D3 (f d1) (f d2) (f d3)
  fmap f (D4 d1 d2 d3 d4) = D4 (f d1) (f d2) (f d3) (f d4)

data TensorDouble_ = TensorDouble_ {
  tdTensor :: !(ForeignPtr CTHDoubleTensor),
  tdDim :: !(TensorDim Word)
  } deriving (Eq, Show)
