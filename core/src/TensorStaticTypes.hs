{-# LANGUAGE DataKinds, KindSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# LANGUAGE MultiParamTypeClasses #-}

module TensorStaticTypes (
  dispS,
  mkT,
  TDS(..)
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

import Data.Proxy (Proxy(..))
import Data.Proxy(Proxy)

{- Version 1: type level # dimensions -}

dispS tensor =
  (withForeignPtr(tdsTensor tensor) dispRaw)

class StaticTensor t where
  mkT :: TensorDim Word -> t

instance (KnownNat n) => StaticTensor (TDS n) where
  mkT dims = unsafePerformIO $ do
    newPtr <- go dims
    fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
    withForeignPtr fPtr fillRaw0
    let n = natVal (Proxy :: Proxy n)
    case dims of
      D0 -> if n == 0 then pure () else fail "Incorrect Dimensions"
      D1 _ -> if n == 1 then pure () else fail "Incorrect Dimensions"
      D2 _ _ -> if n == 2 then pure () else fail "Incorrect Dimensions"
      D3 _ _ _ -> if n == 3 then pure () else fail "Incorrect Dimensions"
      D4 _ _ _ _ -> if n == 4 then pure () else fail "Incorrect Dimensions"
    pure $ makeStatic dims fPtr
    where
      w2cl = fromIntegral -- convert word to CLong
      go D0 = c_THDoubleTensor_new
      go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
      go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                      (w2cl d1) (w2cl d2)
      go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                         (w2cl d1) (w2cl d2) (w2cl d3)
      go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                            (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)
      makeStatic dims fptr = (TDS fptr dims) :: TDS n

data TDS (n :: Nat) = TDS {
  tdsTensor :: !(ForeignPtr CTHDoubleTensor),
  tdsDim :: TensorDim Word
  } deriving (Show, Generic)

testStatic = do
  let foo = (mkT (D2 2 2)) :: TDS 2
  dispS foo -- passes
  let bar = (mkT (D2 2 2)) :: TDS 3
  dispS bar -- fails
  pure ()

{- Version 2: type-level # dimensions + dimension sizes -}

data TDS' (n :: Nat) (d0 :: Nat) (d1 :: Nat) (d2 :: Nat) (d3 :: Nat) = TDS' {
  tdsTensor' :: !(ForeignPtr CTHDoubleTensor),
  tdsDim' :: TensorDim Word
  } deriving (Show, Generic)

instance (KnownNat n, KnownNat d0, KnownNat d1, KnownNat d2, KnownNat d3) => StaticTensor (TDS' n d0 d1 d2 d3) where
  mkT dims = unsafePerformIO $ do
    newPtr <- go dims
    fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
    withForeignPtr fPtr fillRaw0
    let n = natVal (Proxy :: Proxy n)
    case dims of
      D0 -> if n == 0 then pure () else fail "Incorrect Dimensions"
      D1 _ -> if n == 1 then pure () else fail "Incorrect Dimensions"
      D2 _ _ -> if n == 2 then pure () else fail "Incorrect Dimensions"
      D3 _ _ _ -> if n == 3 then pure () else fail "Incorrect Dimensions"
      D4 _ _ _ _ -> if n == 4 then pure () else fail "Incorrect Dimensions"
    pure $ makeStatic dims fPtr
    where
      w2cl = fromIntegral -- convert word to CLong
      go D0 = c_THDoubleTensor_new
      go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
      go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                      (w2cl d1) (w2cl d2)
      go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                         (w2cl d1) (w2cl d2) (w2cl d3)
      go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                            (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)
      makeStatic dims fptr = (TDS' fptr dims) :: TDS' n d0 d1 d2 d3

{- Version 3: type-level # dimensions + dimension sizes as tuple-}

data TDS'' (n :: Nat) (d :: (Nat, Nat, Nat, Nat)) = TDS'' {
  tdsTensor'' :: !(ForeignPtr CTHDoubleTensor),
  tdsDim'' :: TensorDim Word
  } deriving (Show, Generic)

-- instance (KnownNat n, KnownNat d0, KnownNat d1, KnownNat d2, KnownNat d3) => StaticTensor (TDS'' n (d0, d1, d2, d3)) where
--   mkT dims = unsafePerformIO $ do
--     newPtr <- go dims
--     fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
--     withForeignPtr fPtr fillRaw0
--     let n = natVal (Proxy :: Proxy n)
--     case dims of
--       D0 -> if n == 0 then pure () else fail "Incorrect Dimensions"
--       D1 _ -> if n == 1 then pure () else fail "Incorrect Dimensions"
--       D2 _ _ -> if n == 2 then pure () else fail "Incorrect Dimensions"
--       D3 _ _ _ -> if n == 3 then pure () else fail "Incorrect Dimensions"
--       D4 _ _ _ _ -> if n == 4 then pure () else fail "Incorrect Dimensions"
--     pure $ makeStatic dims fPtr
--     where
--       w2cl = fromIntegral -- convert word to CLong
--       go D0 = c_THDoubleTensor_new
--       go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
--       go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
--                       (w2cl d1) (w2cl d2)
--       go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
--                          (w2cl d1) (w2cl d2) (w2cl d3)
--       go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
--                             (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)
--       makeStatic dims fptr = (TDS'' fptr dims) :: TDS'' n (d0, d1, d2, d3)

