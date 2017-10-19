{-# LANGUAGE DataKinds, KindSignatures #-}
{-# LANGUAGE DeriveGeneric #-}

module TensorStaticTypes (
                         ) where


import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import THTypes
import THDoubleTensor

import GHC.TypeLits
import GHC.Generics (Generic)

newtype Dim (n :: Nat) t = Dim t
  deriving (Show, Generic)

-- newtype VR n = VR (Dim n (Vector ℝ))
--   deriving (Num,Fractional,Floating,Generic,Binary)

data TensorDoubleStatic = TD {
  tdTensor :: !(ForeignPtr CTHDoubleTensor)
  -- tdDim :: !(TensorDim Word)
  } deriving (Eq, Show)
