{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE InstanceSigs, RankNTypes, PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeInType #-}
module Torch.Core.Tensor.Dynamic.Double
  ( TensorDouble(..)
  , DynamicTH(..)
  , shapeList
  , rank
  -- These don't need to be exported, but concrete types help with tests
  , td_fromListNd
  , td_fromList1d
  , td_resize
  , td_get
  , td_newWithTensor
  , td_new
  , td_new_
  , td_free_
  , td_init
  , td_transpose
  , td_trans
  , td_shape
  ) where

import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Dim (Dim(..), SomeDims(..), someDimsM)
import Torch.Core.Tensor.Types (TensorDouble(..), THForeignRef(getForeign))
import Torch.Raw.Internal (CTHDoubleTensor, CTHLongTensor)
import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified Torch.Raw.Tensor.Generic as Gen
import qualified Torch.Core.Tensor.Dim as Dim


instance IsList TensorDouble where
  type Item TensorDouble = Double
  fromList = td_fromList1d
  toList td = unsafePerformIO $ withForeignPtr (getForeign td) (pure . fmap realToFrac . Gen.flatten)
  {-# NOINLINE toList #-}


class IsList t => DynamicTH t where
  printTensor :: t -> IO ()
  fromListNd :: SomeDims -> [Item t] -> t
  fromList1d :: [Item t] -> t
  resize :: t -> SomeDims -> t
  get :: SomeDims -> t -> Item t
  newWithTensor :: t -> t
  new :: SomeDims -> t
  new_ :: SomeDims -> IO t
  free_ :: t -> IO ()
  init :: SomeDims -> Item t -> t
  transpose :: Word -> Word -> t -> t
  trans :: t -> t
  shape :: t -> SomeDims


instance DynamicTH TensorDouble where
  printTensor = td_p
  fromListNd (SomeDims d) = td_fromListNd d
  fromList1d = td_fromList1d
  resize t (SomeDims d) = td_resize t d
  get (SomeDims d) = td_get d
  newWithTensor = td_newWithTensor
  new (SomeDims d) = td_new d
  new_ (SomeDims d) = td_new_ d
  free_ = td_free_
  init (SomeDims d) = td_init d
  transpose = td_transpose
  trans = td_trans
  shape = td_shape

td_p :: TensorDouble -> IO ()
td_p t = withForeignPtr (getForeign t) Gen.dispRawRealFloat

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME(stites): This should go in MonadThrow
td_fromListNd :: k ~ Nat => Dim (d::[k]) -> [Double] -> TensorDouble
td_fromListNd d l =
  if fromIntegral (product (Dim.dimVals d)) == length l
  then td_resize (td_fromList1d l) d
  else error "Incorrect tensor dimensions specified."

-- |Initialize a 1D tensor from a list
td_fromList1d :: [Double] -> TensorDouble
td_fromList1d l = unsafePerformIO $ do
  sdims <- someDimsM [length l]
  let res = td_new' sdims
  mapM_ (mutTensor (getForeign res)) (zip [0..length l - 1] l)
  pure res
 where
  mutTensor :: ForeignPtr CTHDoubleTensor -> (Int, Double) -> IO ()
  mutTensor t (idx, value) = withForeignPtr t $ \tp ->
    Gen.c_set1d tp (fromIntegral idx) (realToFrac value)
{-# NOINLINE td_fromList1d #-}

-- |Copy contents of tensor into a new one of specified size
td_resize :: k ~ Nat => TensorDouble -> Dim (d::[k]) -> TensorDouble
td_resize t d = unsafePerformIO $ do
  let resDummy = td_new d
  newPtr <- withForeignPtr (getForeign t) Gen.c_newClone
  newFPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  withForeignPtr newFPtr (withForeignPtr (getForeign resDummy) . Gen.c_resizeAs)
  pure $ TensorDouble newFPtr
{-# NOINLINE td_resize #-}

td_get :: Dim (d::[k]) -> TensorDouble -> Double
td_get loc tensor = unsafePerformIO $ withForeignPtr
  (getForeign tensor)
  (\t -> pure . realToFrac $ t `Gen.genericGet` loc)
{-# NOINLINE td_get #-}

td_newWithTensor :: TensorDouble -> TensorDouble
td_newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) Gen.c_newWithTensor
  newFPtr <- newForeignPtr Gen.p_free newPtr
  -- ds <- someDimsM (Gen.dimList newPtr)
  pure $ TensorDouble newFPtr
{-# NOINLINE td_newWithTensor #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new :: k ~ Nat => Dim (d::[k]) -> TensorDouble
td_new dims = unsafePerformIO $ do
  newPtr <- Gen.constant dims 0.0
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ TensorDouble fPtr -- (SomeDims dims)
{-# NOINLINE td_new #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new' :: SomeDims -> TensorDouble
td_new' sdims = unsafePerformIO $ do
  newPtr <- Gen.constant' sdims 0
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ TensorDouble fPtr -- sdims
{-# NOINLINE td_new' #-}


-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new_ :: k ~ Nat => Dim (d::[k]) -> IO TensorDouble
td_new_ ds = do
  newPtr <- Gen.constant ds 0.0
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ TensorDouble fPtr -- (SomeDims ds)

td_free_ :: TensorDouble -> IO ()
td_free_ t = finalizeForeignPtr $! getForeign t

td_init :: k ~ Nat => Dim (d::[k]) -> Double -> TensorDouble
td_init ds val = unsafePerformIO $ do
  newPtr <- Gen.constant ds (realToFrac val)
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  withForeignPtr fPtr (Gen.inplaceFill realToFrac val)
  pure $ TensorDouble fPtr -- (SomeDims ds)
{-# NOINLINE td_init #-}

td_transpose :: Word -> Word -> TensorDouble -> TensorDouble
td_transpose dim1 dim2 t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p dim1C dim2C)
  newFPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  -- ds <- someDimsM (Gen.dimList newPtr)
  pure $ TensorDouble newFPtr -- ds
 where
  dim1C, dim2C :: CInt
  dim1C = fromIntegral dim1
  dim2C = fromIntegral dim2
{-# NOINLINE td_transpose #-}

td_trans :: TensorDouble -> TensorDouble
td_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p 1 0)
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ TensorDouble newFPtr
{-# NOINLINE td_trans #-}

td_shape :: TensorDouble -> SomeDims
td_shape t = unsafePerformIO $ withForeignPtr (getForeign t) (pure . Gen.getDynamicDim)
{-# NOINLINE td_shape #-}

shapeList :: DynamicTH t => t -> [Int]
shapeList = Dim.dimVals' . shape

rank :: DynamicTH t => t -> Int
rank = Dim.rank' . shape
