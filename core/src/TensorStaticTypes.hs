{-# LANGUAGE DataKinds, KindSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}



module TensorStaticTypes (
                         ) where


import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import TensorRaw
import TensorDouble
import TensorTypes
import THTypes
import THDoubleTensor

import GHC.TypeLits
import GHC.Generics (Generic)
import System.IO.Unsafe (unsafePerformIO)

newtype Dim (n :: Nat) t = Dim t
  deriving (Show, Generic)

newtype VR n = VR (Dim n TensorDouble)

test :: IO ()
test = do
  let t = tdNew D0
  let foo = (Dim t) :: Dim 0 TensorDouble
  pure ()

-- foo = VR (Dim 3 undefined)

-- newtype R n = R (Dim n (Vector ℝ))
--   deriving (Num,Fractional,Floating,Generic,Binary)

-- data TensorDoubleStatic = TDS {
--   tdTensor :: !(ForeignPtr CTHDoubleTensor)
--   tdDim :: !(TensorDim Word)
--   } deriving (Eq, Show)

tds = do
  newPtr <- c_THDoubleTensor_new
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr

-- data IntBool a where
--   Int :: Int -> IntBool Int
--   Bool :: Bool -> IntBool Bool

-- |Create a new (double) tensor of specified dimensions and fill it with 0
-- tdS :: Dim n -> TensorDoubleStatic
tdS dims = unsafePerformIO $ do
  newPtr <- go dims
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ makeT dims fPtr
  where
    w2cl = fromIntegral -- word 2 clong
    go D0 = c_THDoubleTensor_new
    go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
    go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                    (w2cl d1) (w2cl d2)
    go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                       (w2cl d1) (w2cl d2) (w2cl d3)
    go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                          (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

-- data MixDim (n :: Nat) a where
--   Dim 0 TensorDouble -> MixDim Dim 0 TensorDouble
  -- Dim 1 TensorDouble -> MixDim Dim 1 TensorDouble
  -- Dim 2 TensorDouble -> MixDim Dim 2 TensorDouble
  -- Dim 3 TensorDouble -> MixDim Dim 3 TensorDouble
  -- Dim 4 TensorDouble -> MixDim Dim 4 TensorDouble

data DimEnc0
data DimEnc1
data DimEnc2
data DimEnc3
data DimEnc4

-- type family MixDim a where
--     MixDim DimEnc0 = Dim 0
--     MixDim DimEnc1 = Dim 1
--     MixDim DimEnc2 = Dim 2
--     MixDim DimEnc3 = Dim 3
--     MixDim DimEnc4 = Dim 4

type family MixDim a where
    MixDim (Dim 0) = Dim 0
    MixDim (Dim 1) = Dim 1
    MixDim (Dim 2) = Dim 2
    MixDim (Dim 3) = Dim 3
    MixDim (Dim 4) = Dim 4

-- TODO how to compute the type signature of Dim this?
makeT :: TensorDim Word -> ForeignPtr CTHDoubleTensor -> Dim n TensorDouble
makeT dims@D0 fptr = Dim $ TensorDouble fptr dims -- :: Dim 0 TensorDouble
makeT dims@(D1 _) fptr = Dim $ TensorDouble fptr dims -- :: Dim 1 TensorDouble
makeT dims@(D2 _ _) fptr =  Dim $ TensorDouble fptr dims -- :: Dim 2 TensorDouble
makeT dims@(D3 _ _ _) fptr =  Dim $ TensorDouble fptr dims -- :: Dim 3 TensorDouble
makeT dims@(D4 _ _ _ _) fptr =  Dim $ TensorDouble fptr dims -- :: Dim 4 TensorDouble
