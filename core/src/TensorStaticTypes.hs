{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorStaticTypes (
  mkT,
  dispS,
  TDS(..),
  TDS'(..)
  ) where

import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )

import TensorRaw
import TensorDouble
import TensorTypes
import THTypes
import THDoubleTensor

import GHC.TypeLits
import System.IO.Unsafe (unsafePerformIO)

import Data.Proxy (Proxy(..))
import Data.Proxy(Proxy)

{- Version 1: type level # dimensions -}

class StaticTensor t where
  mkT :: TensorDim Word -> t
  dispS :: t -> IO ()

w2cl :: Word -> CLong
w2cl = fromIntegral

mkPtr :: TensorDim Word -> IO (Ptr CTHDoubleTensor)
mkPtr D0 = c_THDoubleTensor_new
mkPtr (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
mkPtr (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                   (w2cl d1) (w2cl d2)
mkPtr (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                      (w2cl d1) (w2cl d2) (w2cl d3)
mkPtr (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                         (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

-- Runtime type-level check
dimCheck :: Monad m => TensorDim Word -> Integer -> m ()
dimCheck dims n =
  case dims of
    D0 -> if n == 0 then pure () else fail "Incorrect Dimensions"
    D1 _ -> if n == 1 then pure () else fail "Incorrect Dimensions"
    D2 _ _ -> if n == 2 then pure () else fail "Incorrect Dimensions"
    D3 _ _ _ -> if n == 3 then pure () else fail "Incorrect Dimensions"
    D4 _ _ _ _ -> if n == 4 then pure () else fail "Incorrect Dimensions"

instance (KnownNat n) => StaticTensor (TDS n) where
  mkT dims = unsafePerformIO $ do
    newPtr <- mkPtr dims
    fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
    withForeignPtr fPtr fillRaw0
    let n = natVal (Proxy :: Proxy n)
    dimCheck dims n
    pure $ makeStatic dims fPtr
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS n
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

data TDS (n :: Nat) = TDS {
  tdsTensor :: !(ForeignPtr CTHDoubleTensor),
  tdsDim :: TensorDim Word
  } deriving (Show)

testStatic = do
  let foo = (mkT (D2 2 2)) :: TDS 2
  dispS foo -- passes
  let bar = (mkT (D2 2 2)) :: TDS 3
  dispS bar -- fails
  pure ()

{- Version 2: type-level # dimensions + dimension sizes -}

data TDS' (n :: Nat) (d :: (Nat, Nat, Nat, Nat)) = TDS' {
  tdsTensor' :: !(ForeignPtr CTHDoubleTensor),
  tdsDim' :: TensorDim Word
  } deriving (Show)

instance (KnownNat n, KnownNat d0, KnownNat d1, KnownNat d2, KnownNat d3) =>
  StaticTensor (TDS' n '(d0 , d1 , d2 , d3) )  where
  mkT dims = unsafePerformIO $ do
    newPtr <- mkPtr dims
    fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
    withForeignPtr fPtr fillRaw0
    let n = natVal (Proxy :: Proxy n)
    dimCheck dims n
    pure $ makeStatic dims fPtr
    where
      makeStatic dims fptr = (TDS' fptr dims) :: TDS' n '(d0, d1, d2, d3)
  dispS tensor = (withForeignPtr(tdsTensor' tensor) dispRaw)

testStatic2 = do
  let foo = (mkT (D2 2 2)) :: TDS' 2 '(2, 2, 0, 0)
  dispS foo -- passes
  let bar = (mkT (D2 2 2)) :: TDS' 3 '(2, 2, 0, 0)
  dispS bar -- fails
  pure ()

test = do
  testStatic
  testStatic2
