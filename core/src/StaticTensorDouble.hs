{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}


module StaticTensorDouble (
  mkT,
  dispS,
  TensorDoubleStatic(..),
  TDS(..),
  Nat -- re-export for kind signature readability
  ) where

import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )

import TensorRaw
import TensorDouble
import TensorTypes
import THTypes
import THDoubleTensor
import THDoubleTensorMath

import GHC.TypeLits (Nat, KnownNat, natVal)
import System.IO.Unsafe (unsafePerformIO)

import Data.Proxy (Proxy(..))
import Data.Proxy (Proxy)

class StaticTensor t where
  -- |create tensor
  mkT :: t
  -- |create and initialize tensor
  mkTInit :: Double -> t
  -- |Display tensor
  dispS :: t -> IO ()

-- |Convert word to CLong
w2cl :: Word -> CLong
w2cl = fromIntegral

-- |Runtime type-level check of # dimensions
dimCheck :: Monad m => TensorDim Word -> Integer -> m ()
dimCheck dims n = case dims of
  D0 -> if n == 0 then pure () else fail "Incorrect Dimensions"
  D1 _ -> if n == 1 then pure () else fail "Incorrect Dimensions"
  D2 _ _ -> if n == 2 then pure () else fail "Incorrect Dimensions"
  D3 _ _ _ -> if n == 3 then pure () else fail "Incorrect Dimensions"
  D4 _ _ _ _ -> if n == 4 then pure () else fail "Incorrect Dimensions"

data TensorDoubleStatic (n :: Nat) (d :: [Nat]) = TDS {
  tdsTensor :: !(ForeignPtr CTHDoubleTensor),
  tdsDim :: TensorDim Word
  } deriving (Show)

type TDS = TensorDoubleStatic

-- |Make a low level pointer according to dimensions
mkPtr dim value = tensorRaw dim value

mkTHelper dims ndim makeStatic value = unsafePerformIO $ do
  newPtr <- mkPtr dims value
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  -- dimCheck dims ndim
  pure $ makeStatic dims fPtr

instance Eq (TensorDoubleStatic n d) where
  (==) t1 t2 = unsafePerformIO $ withForeignPtr (tdsTensor t1)
    (\t1c ->
        withForeignPtr (tdsTensor t2)
          (\t2c -> pure $ (c_THDoubleTensor_equal t1c t2c) == 1
          )
    )

-- instance StaticTensor (TensorDoubleStatic n d) where
--   dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance (KnownNat d0, KnownNat d1, KnownNat d2, KnownNat d3) =>
  StaticTensor (TensorDoubleStatic 4 '[d0, d1, d2, d3] )  where
  mkTInit initVal = mkTHelper dims 4 makeStatic initVal
    where
      dims = D4 s0 s1 s2 s3
      makeStatic dims fptr = (TDS fptr dims) :: TDS 4 '[d0, d1, d2, d3]
      [s0, s1, s2, s3] = fromIntegral <$>
                         [natVal (Proxy :: Proxy d0), natVal (Proxy :: Proxy d1),
                          natVal (Proxy :: Proxy d2), natVal (Proxy :: Proxy d3)]
  mkT = mkTInit 0.0
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance (KnownNat d0, KnownNat d1, KnownNat d2) =>
  StaticTensor (TensorDoubleStatic 3 '[d0, d1, d2] )  where
  mkTInit initVal = mkTHelper dims 3 makeStatic initVal
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 3 '[d0, d1, d2]
      [s0, s1, s2] = fromIntegral <$>
                     [natVal (Proxy :: Proxy d0), natVal (Proxy :: Proxy d1),
                      natVal (Proxy :: Proxy d2)]
      dims = D3 s0 s1 s2
  mkT = mkTInit 0.0
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance (KnownNat d0, KnownNat d1) =>
  StaticTensor (TensorDoubleStatic 2 '[d0, d1] )  where
  mkTInit initVal = mkTHelper dims 2 makeStatic initVal
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 2 '[d0, d1]
      [s0, s1] = fromIntegral <$>
                 [natVal (Proxy :: Proxy d0), natVal (Proxy :: Proxy d1)]
      dims = D2 s0 s1
  mkT = mkTInit 0.0
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)


instance (KnownNat d0) =>
  StaticTensor (TensorDoubleStatic 1 '[d0] )  where
  mkTInit initVal = mkTHelper dims 1 makeStatic initVal
    where
      makeStatic dims fptr = (TDS fptr dims) :: TDS 1 '[d0]
      s0 = fromIntegral $ natVal (Proxy :: Proxy d0)
      dims = D1 s0
  mkT = mkTInit 0.0
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

instance StaticTensor (TensorDoubleStatic 0 '[] )  where
  mkTInit initVal = mkTHelper dims 0 makeStatic initVal
    where
      dims = D0
      makeStatic dims fptr = (TDS fptr dims) :: TDS 0 '[]
  mkT = mkTInit 0.0
  dispS tensor = (withForeignPtr(tdsTensor tensor) dispRaw)

{- Sanity checks -}

testStatic = do
  print("1")
  let t1 = mkT :: TDS 2 '[2, 2]
  dispS t1 -- passes
  print("2")
  let t2 = mkT :: TDS 2 '[2, 4] -- should fail but doesn't yet
  dispS t2
  print("3")
  let t3 = mkT :: TDS 3 '[2, 2, 2] -- fails due to dim mismatch
  dispS t3
  print("4")
  let t4 = mkT :: TDS 2 '[8, 4] -- fails due to dim mismatch
  dispS t4
  pure ()

testEq = do
  print "Should be True:"
  print $ (mkTInit 4.0 :: TDS 2 '[2,3]) ==  (mkTInit 4.0 :: TDS 2 '[2,3])
  print "Should be False:"
  print $ (mkTInit 3.0 :: TDS 2 '[2,3]) ==  (mkTInit 1.0 :: TDS 2 '[2,3])

test = do
  testStatic
  testEq
