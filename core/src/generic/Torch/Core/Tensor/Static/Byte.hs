{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Core.Tensor.Static.Byte (
  tbs_new,
  tbs_cloneDim,
  tbs_init,
  tbs_p,
  TBS,
  TensorByteStatic(..)
  ) where

import Data.Singletons
-- import Data.Singletons.Prelude
import Data.Singletons.TypeLits

import Foreign (Ptr)
import Foreign.C.Types (CLLong)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dim
import Torch.Raw.Tensor.Generic (genericNew)
import THByteTensor
import THByteTensorMath
import THTypes
import qualified Torch.Raw.Tensor.Generic as Gen

class TBClass t where
  -- |tensor dimensions
  -- |create tensor
  tbs_new :: t
  -- |create tensor of the same dimensions
  tbs_cloneDim :: t -> t -- takes unused argument, gets dimensions by matching types
  -- |create and initialize tensor
  tbs_init :: Int -> t
  -- |Display tensor
  tbs_p ::  t -> IO ()

newtype TensorByteStatic (d :: [Nat])
  = TBS
  { tbsTensor :: ForeignPtr CTHByteTensor
  } deriving (Show)

type TBS = TensorByteStatic

-- |Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
-- fillRaw :: Real a => a -> TensorByteRaw -> IO ()
fillRaw value = (flip c_THByteTensor_fill) (fromIntegral value)

-- | Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
tensorRaw :: Dim (ns::[Nat]) -> Int -> IO TensorByteRaw
tensorRaw dims value = do
  newPtr <- genericNew dims
  Gen.fillZeros newPtr

{-
list2dim :: (Num a2, Integral a1) => [a1] -> TensorDim a2
list2dim lst  = case (length lst) of
  0 -> D0
  1 -> D1 (d !! 0)
  2 -> D2 ((d !! 0), (d !! 1))
  3 -> D3 ((d !! 0), (d !! 1), (d !! 2))
  4 -> D4 ((d !! 0), (d !! 1), (d !! 2), (d !! 3))
  _ -> error "Tensor type signature has invalid dimensions"
  where
    d = fromIntegral <$> lst -- cast as needed for tensordim
-}

-- |Make a foreign pointer from requested dimensions
mkTHelper
  :: Dim (ns::[Nat])
  -> (Dim (ns::[Nat]) -> ForeignPtr CTHByteTensor -> a)
  -> Int
  -> a
mkTHelper dims makeStatic value = unsafePerformIO $ do
  newPtr <- Gen.constant dims (fromIntegral value)
  fPtr <- newForeignPtr Gen.p_free newPtr
  pure $ makeStatic dims fPtr
{-# NOINLINE mkTHelper #-}

instance SingI d => TBClass (TensorByteStatic (d :: [Nat]))  where
  tbs_init initVal = mkTHelper dims makeStatic initVal
    where
      dims = undefined -- list2dim $ fromSing (sing :: Sing d)
      makeStatic dims fptr = (TBS fptr) :: TBS d
  tbs_new = tbs_init 0
  tbs_cloneDim _ = tbs_new :: TBS d
  tbs_p tensor = (withForeignPtr (tbsTensor tensor) Gen.dispRaw)


