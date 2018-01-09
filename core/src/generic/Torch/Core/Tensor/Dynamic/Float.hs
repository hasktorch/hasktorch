{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE InstanceSigs, RankNTypes, PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeInType #-}
module Torch.Core.Tensor.Dynamic.Float
  ( TensorFloat(..)
  ) where

import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)

import Torch.Core.Tensor.Dim (Dim(..), SomeDims(..), someDimsM)
import Torch.Core.Tensor.Types (TensorFloat(..), THForeignRef(..))
import Torch.Raw.Internal (CTHDoubleTensor, CTHLongTensor)
import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified Torch.Raw.Tensor.Generic as GenRaw
import qualified Torch.Core.Tensor.Dynamic.Generic as Gen
import qualified Torch.Core.Tensor.Dim as Dim

import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Types
import THTypes

instance IsList TensorFloat where
  type Item TensorFloat = Float
  fromList = td_fromList1d
  toList td = unsafePerformIO $ withForeignPtr (getForeign td) (pure . fmap realToFrac . GenRaw.flatten)
  {-# NOINLINE toList #-}


class IsList t => THTensor t where
  printTensor :: t -> IO ()
  fromListNd :: k ~ Nat => Dim (d::[k]) -> [Item t] -> t
  fromList1d :: [Item t] -> t
  resize :: k ~ Nat => t -> Dim (d::[k]) -> t
  get :: k ~ Nat => Dim (d::[k]) -> t -> Item t
  newWithTensor :: t -> t
  new :: k ~ Nat => Dim (d::[k]) -> t
  new_ :: k ~ Nat => Dim (d::[k]) -> IO t
  free_ :: t -> IO ()
  init :: k ~ Nat => Dim (d::[k]) -> Item t -> t
  transpose :: Word -> Word -> t -> t
  trans :: t -> t


instance THTensor TensorFloat where
  printTensor = td_p
  fromListNd = td_fromListNd
  fromList1d = td_fromList1d
  resize = td_resize
  get = td_get
  newWithTensor = td_newWithTensor
  new = td_new
  new_ = td_new_
  free_ = td_free_
  init = td_init
  transpose = td_transpose
  trans = td_trans

td_p :: TensorFloat -> IO ()
td_p t = withForeignPtr (getForeign t) GenRaw.dispRaw

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME(stites): This should go in MonadThrow
td_fromListNd :: k ~ Nat => Dim (d::[k]) -> [Float] -> TensorFloat
td_fromListNd d l =
  if fromIntegral (product (Dim.dimVals d)) == length l
  then td_resize (td_fromList1d l) d
  else error "Incorrect tensor dimensions specified."

-- |Initialize a 1D tensor from a list
td_fromList1d :: [Float] -> TensorFloat
td_fromList1d l = unsafePerformIO $ do
  sdims <- someDimsM [length l]
  let res = td_new' sdims
  mapM_ (mutTensor (getForeign res)) (zip [0..length l - 1] l)
  pure res
 where
  mutTensor :: ForeignPtr CTHFloatTensor -> (Int, Float) -> IO ()
  mutTensor t (idx, value) = withForeignPtr t $ \tp ->
    GenRaw.c_set1d tp (fromIntegral idx) (realToFrac value)
{-# NOINLINE td_fromList1d #-}

-- |Copy contents of tensor into a new one of specified size
td_resize :: k ~ Nat => TensorFloat -> Dim (d::[k]) -> TensorFloat
td_resize t d = unsafePerformIO $ do
  let resDummy = td_new d
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  withForeignPtr newFPtr (withForeignPtr (getForeign resDummy) . GenRaw.c_resizeAs)
  pure $ TensorFloat newFPtr
{-# NOINLINE td_resize #-}

td_get :: Dim (d::[k]) -> TensorFloat -> Float
td_get loc tensor = unsafePerformIO $ withForeignPtr
  (getForeign tensor)
  (\t -> pure . realToFrac $ t `GenRaw.genericGet` loc)
{-# NOINLINE td_get #-}

td_newWithTensor :: TensorFloat -> TensorFloat
td_newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newWithTensor
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  -- ds <- someDimsM (Gen.dimList newPtr)
  pure $ TensorFloat newFPtr
{-# NOINLINE td_newWithTensor #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new :: k ~ Nat => Dim (d::[k]) -> TensorFloat
td_new dims = unsafePerformIO $ do
  newPtr <- GenRaw.constant dims 0.0
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  void $ withForeignPtr fPtr GenRaw.fillZeros
  pure $ TensorFloat fPtr -- (SomeDims dims)
{-# NOINLINE td_new #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new' :: SomeDims -> TensorFloat
td_new' sdims = unsafePerformIO $ do
  newPtr <- GenRaw.constant' sdims 0
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  void $ withForeignPtr fPtr GenRaw.fillZeros
  pure $ TensorFloat fPtr -- sdims
{-# NOINLINE td_new' #-}


-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new_ :: k ~ Nat => Dim (d::[k]) -> IO TensorFloat
td_new_ ds = do
  newPtr <- GenRaw.constant ds 0.0
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  void $ withForeignPtr fPtr GenRaw.fillZeros
  pure $ TensorFloat fPtr -- (SomeDims ds)

td_free_ :: TensorFloat -> IO ()
td_free_ t = finalizeForeignPtr $! getForeign t

td_init :: k ~ Nat => Dim (d::[k]) -> Float -> TensorFloat
td_init ds val = unsafePerformIO $ do
  newPtr <- GenRaw.constant ds (realToFrac val)
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  withForeignPtr fPtr (GenRaw.inplaceFill realToFrac val)
  pure $ TensorFloat fPtr -- (SomeDims ds)
{-# NOINLINE td_init #-}

td_transpose :: Word -> Word -> TensorFloat -> TensorFloat
td_transpose dim1 dim2 t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> GenRaw.c_newTranspose p dim1C dim2C)
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  -- ds <- someDimsM (Gen.dimList newPtr)
  pure $ TensorFloat newFPtr -- ds
 where
  dim1C, dim2C :: CInt
  dim1C = fromIntegral dim1
  dim2C = fromIntegral dim2
{-# NOINLINE td_transpose #-}

td_trans :: TensorFloat -> TensorFloat
td_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> GenRaw.c_newTranspose p 1 0)
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ TensorFloat newFPtr
{-# NOINLINE td_trans #-}


