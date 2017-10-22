{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}

module StaticTensorDouble (
  mkT,
  dispS,
  TensorDoubleStatic(..),
  TensorDoubleStatic'(..),
  TensorDoubleStatic''(..),
  TDS(..),
  TDS'(..),
  TDS''(..)
  ) where

import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )

import TensorRaw
import TensorDouble
import TensorTypes
import THTypes
import THDoubleTensor

import GHC.TypeLits (Nat, KnownNat, natVal)
import System.IO.Unsafe (unsafePerformIO)

import Data.Proxy (Proxy(..))
import Data.Proxy(Proxy)

class StaticTensor t where
  -- |create tensor
  mkT :: TensorDim Word -> t
  -- |Display tensor
  dispS :: t -> IO ()

w2cl :: Word -> CLong
w2cl = fromIntegral

-- |Make a low level pointer according to dimensions
mkPtr dim = tensorRaw dim 0.0

-- |Runtime type-level check of # dimensions
dimCheck :: Monad m => TensorDim Word -> Integer -> m ()
dimCheck dims n = case dims of
  D0 -> if n == 0 then pure () else fail "Incorrect Dimensions"
  D1 _ -> if n == 1 then pure () else fail "Incorrect Dimensions"
  D2 _ _ -> if n == 2 then pure () else fail "Incorrect Dimensions"
  D3 _ _ _ -> if n == 3 then pure () else fail "Incorrect Dimensions"
  D4 _ _ _ _ -> if n == 4 then pure () else fail "Incorrect Dimensions"

{- Version 1: list representation of sizes -}

data TensorDoubleStatic (n :: Nat) (d :: [Nat]) = TDS {
  tdsTensor :: !(ForeignPtr CTHDoubleTensor),
  tdsDim :: TensorDim Word
  } deriving (Show)

type TDS = TensorDoubleStatic

mkTHelper dims ndim makeStatic = unsafePerformIO $ do
  newPtr <- mkPtr dims
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  dimCheck dims ndim
  pure $ makeStatic dims fPtr

instance (KnownNat d0, KnownNat d1, KnownNat d2, KnownNat d3) =>
  StaticTensor (TensorDoubleStatic 4 '[d0, d1, d2, d3] )  where
  mkT dims = mkTHelper dims 4 makeStatic
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 4 '[d0, d1, d2, d3]
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance (KnownNat d0, KnownNat d1, KnownNat d2) =>
  StaticTensor (TensorDoubleStatic 3 '[d0, d1, d2] )  where
  mkT dims = mkTHelper dims 3 makeStatic
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 3 '[d0, d1, d2]
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance (KnownNat d0, KnownNat d1) =>
  StaticTensor (TensorDoubleStatic 2 '[d0, d1] )  where
  mkT dims = mkTHelper dims 2 makeStatic
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 2 '[d0, d1]
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance (KnownNat d0) =>
  StaticTensor (TensorDoubleStatic 1 '[d0] )  where
  mkT dims = mkTHelper dims 1 makeStatic
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 1 '[d0]
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance StaticTensor (TensorDoubleStatic 0 '[] )  where
  mkT dims = mkTHelper dims 0 makeStatic
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 0 '[]
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

{- Version 2: type-level # dimensions + dimension sizes as tuple -}

data TensorDoubleStatic' (n :: Nat) (d :: (Nat, Nat, Nat, Nat)) = TDS' {
  tdsTensor' :: !(ForeignPtr CTHDoubleTensor),
  tdsDim' :: TensorDim Word
  } deriving (Show)

type TDS' = TensorDoubleStatic'

instance (KnownNat n, KnownNat d0, KnownNat d1, KnownNat d2, KnownNat d3) =>
  StaticTensor (TensorDoubleStatic' n '(d0 , d1 , d2 , d3) )  where
  mkT dims = unsafePerformIO $ do
    newPtr <- mkPtr dims
    fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
    withForeignPtr fPtr fillRaw0
    let n = natVal (Proxy :: Proxy n)
    dimCheck dims n
    pure $ makeStatic dims fPtr
    where
      makeStatic dims fptr = (TDS' fptr dims) :: TensorDoubleStatic' n '(d0, d1, d2, d3)
  dispS tensor = (withForeignPtr(tdsTensor' tensor) dispRaw)

{- Version 3: type level representation for # dimensions only -}

data TensorDoubleStatic'' (n :: Nat) = TDS'' {
  tdsTensor'' :: !(ForeignPtr CTHDoubleTensor),
  tdsDim'' :: TensorDim Word
  } deriving (Show)

type TDS'' = TensorDoubleStatic''

instance (KnownNat n) => StaticTensor (TDS'' n) where
  mkT dims = unsafePerformIO $ do
    newPtr <- mkPtr dims
    fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
    withForeignPtr fPtr fillRaw0
    let n = natVal (Proxy :: Proxy n)
    dimCheck dims n
    pure $ makeStatic dims fPtr
    where
      makeStatic dims fptr = (TDS'' fptr dims) :: TDS'' n
  dispS tensor = (withForeignPtr(tdsTensor'' tensor) dispRaw)

{- Sanity checks -}

testStatic = do
  print("1")
  let foo = (mkT (D2 2 2)) :: TDS 2 '[2, 2]
  dispS foo -- passes
  print("2")
  let bar = (mkT (D2 2 2)) :: TDS 2 '[2, 4] -- should fail but doesn't yet
  dispS bar
  print("3")
  let bar = (mkT (D2 2 2)) :: TDS 3 '[2, 2, 2] -- fails due to dim mismatch
  dispS bar
  pure ()

testStatic' = do
  print("1")
  let foo = (mkT (D2 2 2)) :: TDS' 2 '(2, 2, 0, 0)
  dispS foo -- passes
  print("2")
  let bar = (mkT (D2 2 2)) :: TDS' 2 '(2, 4, 0, 0) -- should fail but doesn't yet
  dispS bar
  print("3")
  let bar = (mkT (D2 2 2)) :: TDS' 3 '(2, 2, 2, 0) -- fails due to dim mismatch
  dispS bar -- fails
  pure ()

testStatic'' = do
  let foo = (mkT (D2 2 2)) :: TDS'' 2
  dispS foo -- passes
  let bar = (mkT (D2 2 2)) :: TDS'' 3
  dispS bar -- fails
  pure ()

test = do
  testStatic
  testStatic'
  testStatic''
