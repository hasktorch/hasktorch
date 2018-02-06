{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
module Torch.Core.Tensor.Dynamic
  ( Tensor(..)
  ) where

import Foreign (Ptr, withForeignPtr, newForeignPtr)
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import Torch.Class.Internal (HsReal, HsAccReal, HsStorage)
import THTypes
import qualified Tensor as Sig
import qualified Torch.Class.Tensor as Class

import Torch.Core.Storage (Storage(..), asStorage)

newtype Tensor = Tensor { tensor :: ForeignPtr Sig.CTensor }
  deriving (Show, Eq)

type instance HsReal    Tensor = Sig.CReal
type instance HsAccReal Tensor = Sig.CAccReal
type instance HsStorage Tensor = Storage

asTensor :: Ptr Sig.CTensor -> IO Tensor
asTensor = fmap Tensor . newForeignPtr Sig.p_free

instance Class.IsTensor Tensor where
  tensordata :: Tensor -> IO (Ptr (HsReal Tensor))
  tensordata t = withForeignPtr (tensor t) Sig.c_data

  clearFlag :: Tensor -> CChar -> IO ()
  clearFlag t cc = withForeignPtr (tensor t) $ \t' -> Sig.c_clearFlag t' cc

  desc :: Tensor -> IO CTHDescBuff
  desc t = withForeignPtr (tensor t) (pure . Sig.c_desc)

  expand :: Tensor -> Tensor -> Ptr CTHLongStorage -> IO ()
  expand t0 t1 ls =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_expand t0' t1' ls

  -- expandNd :: Ptr Tensor -> Ptr Tensor -> CInt -> IO ()
  -- expandNd = undefined

  free :: Tensor -> IO ()
  free t = withForeignPtr (tensor t) Sig.c_free

  freeCopyTo :: Tensor -> Tensor -> IO ()
  freeCopyTo t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_freeCopyTo t0' t1'

  get1d :: Tensor -> CLLong -> IO (HsReal Tensor)
  get1d t d1 = withForeignPtr (tensor t) $ \t' -> pure $ Sig.c_get1d t' d1

  get2d :: Tensor -> CLLong -> CLLong -> IO (HsReal Tensor)
  get2d t d1 d2 = withForeignPtr (tensor t) $ \t' -> pure $ Sig.c_get2d t' d1 d2

  get3d :: Tensor -> CLLong -> CLLong -> CLLong -> IO (HsReal Tensor)
  get3d t d1 d2 d3 = withForeignPtr (tensor t) $ \t' -> pure $ Sig.c_get3d t' d1 d2 d3

  get4d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO (HsReal Tensor)
  get4d t d1 d2 d3 d4 = withForeignPtr (tensor t) $ \t' -> pure $ Sig.c_get4d t' d1 d2 d3 d4

  isContiguous :: Tensor -> IO Bool
  isContiguous t =
    withForeignPtr (tensor t) $ \t' ->
      pure $ 1 == Sig.c_isContiguous t'

  isSameSizeAs :: Tensor -> Tensor -> IO Bool
  isSameSizeAs t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        pure $ 1 == Sig.c_isSetTo t0' t1'

  isSetTo :: Tensor -> Tensor -> IO Bool
  isSetTo t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        pure $ 1 == Sig.c_isSetTo t0' t1'

  isSize :: Tensor -> Ptr CTHLongStorage -> IO Bool
  isSize t ls =
    withForeignPtr (tensor t) $ \t' ->
      pure $ 1 == Sig.c_isSize t' ls

  nDimension :: Tensor -> IO CInt
  nDimension t = withForeignPtr (tensor t) (pure . Sig.c_nDimension)

  nElement :: Tensor -> IO CPtrdiff
  nElement t = withForeignPtr (tensor t) (pure . Sig.c_nElement)

  narrow :: Tensor -> Tensor -> CInt -> CLLong -> CLLong -> IO ()
  narrow t0 t1 a b c =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_narrow t0' t1' a b c

  new :: IO Tensor
  new = Sig.c_new >>= asTensor

  newClone :: Tensor -> IO Tensor
  newClone t = withForeignPtr (tensor t) Sig.c_newClone >>= asTensor

  newContiguous :: Tensor -> IO Tensor
  newContiguous t =
    withForeignPtr (tensor t) Sig.c_newContiguous >>= asTensor

  newExpand :: Tensor -> Ptr CTHLongStorage -> IO Tensor
  newExpand t ls =
    withForeignPtr (tensor t) (\t' -> Sig.c_newExpand t' ls) >>= asTensor

  newNarrow :: Tensor -> CInt -> CLLong -> CLLong -> IO Tensor
  newNarrow t a b c =
    withForeignPtr (tensor t) (\t' -> Sig.c_newNarrow t' a b c) >>= asTensor

  newSelect :: Tensor -> CInt -> CLLong -> IO Tensor
  newSelect t a b =
    withForeignPtr (tensor t) (\t' -> Sig.c_newSelect t' a b) >>= asTensor

  newSizeOf :: Tensor -> IO (Ptr CTHLongStorage)
  newSizeOf t = withForeignPtr (tensor t) Sig.c_newSizeOf

  newStrideOf :: Tensor -> IO (Ptr CTHLongStorage)
  newStrideOf t = withForeignPtr (tensor t) Sig.c_newStrideOf

  newTranspose :: Tensor -> CInt -> CInt -> IO Tensor
  newTranspose t a b =
    withForeignPtr (tensor t) (\t' -> Sig.c_newTranspose t' a b) >>= asTensor

  newUnfold :: Tensor -> CInt -> CLLong -> CLLong -> IO Tensor
  newUnfold t a b c =
    withForeignPtr (tensor t) (\t' -> Sig.c_newUnfold t' a b c) >>= asTensor

  newView :: Tensor -> Ptr CTHLongStorage -> IO Tensor
  newView t ls =
    withForeignPtr (tensor t) (\t' -> Sig.c_newView t' ls) >>= asTensor

  newWithSize :: Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO Tensor
  newWithSize ls0 ls1 = Sig.c_newWithSize ls0 ls1 >>= asTensor

  newWithSize1d :: CLLong -> IO Tensor
  newWithSize1d a0 = Sig.c_newWithSize1d a0 >>= asTensor

  newWithSize2d :: CLLong -> CLLong -> IO Tensor
  newWithSize2d a0 a1 = Sig.c_newWithSize2d a0 a1 >>= asTensor

  newWithSize3d :: CLLong -> CLLong -> CLLong -> IO Tensor
  newWithSize3d a0 a1 a2 = Sig.c_newWithSize3d a0 a1 a2 >>= asTensor

  newWithSize4d :: CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithSize4d a0 a1 a2 a3 = Sig.c_newWithSize4d a0 a1 a2 a3 >>= asTensor

  newWithStorage :: HsStorage Tensor -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO Tensor
  newWithStorage s pd ls ls0 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage s' pd ls ls0)
      >>= asTensor

  newWithStorage1d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> IO Tensor
  newWithStorage1d s pd l00 l01 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage1d s' pd l00 l01)
      >>= asTensor


  newWithStorage2d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithStorage2d s pd d00 d01 d10 d11 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage2d s' pd d00 d01 d10 d11)
      >>= asTensor


  newWithStorage3d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithStorage3d s pd d00 d01 d10 d11 d20 d21 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage3d s' pd d00 d01 d10 d11 d20 d21)
      >>= asTensor


  newWithStorage4d :: HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO Tensor
  newWithStorage4d s pd d00 d01 d10 d11 d20 d21 d30 d31 =
    withForeignPtr (storage s) (\s' -> Sig.c_newWithStorage4d s' pd d00 d01 d10 d11 d20 d21 d30 d31)
      >>= asTensor

  newWithTensor :: Tensor -> IO Tensor
  newWithTensor t = withForeignPtr (tensor t) Sig.c_newWithTensor >>= asTensor

  resize :: Tensor -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  resize t l0 l1 = withForeignPtr (tensor t) (\t' -> Sig.c_resize t' l0 l1)

  resize1d :: Tensor -> CLLong -> IO ()
  resize1d t l0 = withForeignPtr (tensor t) (\t' -> Sig.c_resize1d t' l0)

  resize2d :: Tensor -> CLLong -> CLLong -> IO ()
  resize2d t l0 l1 = withForeignPtr (tensor t) (\t' -> Sig.c_resize2d t' l0 l1)
  resize3d :: Tensor -> CLLong -> CLLong -> CLLong -> IO ()
  resize3d t l0 l1 l2 = withForeignPtr (tensor t) (\t' -> Sig.c_resize3d t' l0 l1 l2)
  resize4d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resize4d t l0 l1 l2 l3 = withForeignPtr (tensor t) (\t' -> Sig.c_resize4d t' l0 l1 l2 l3)
  resize5d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  resize5d t l0 l1 l2 l3 l4 = withForeignPtr (tensor t) (\t' -> Sig.c_resize5d t' l0 l1 l2 l3 l4)
  resizeAs :: Tensor -> Tensor -> IO ()
  resizeAs t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_resizeAs t0' t1'

  resizeNd :: Tensor -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  resizeNd t i l0 l1 = withForeignPtr (tensor t) (\t' -> Sig.c_resizeNd t' i l0 l1)

  retain :: Tensor -> IO ()
  retain t = withForeignPtr (tensor t) Sig.c_retain

  select :: Tensor -> Tensor -> CInt -> CLLong -> IO ()
  select t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_select t0' t1' a b

  set :: Tensor -> Tensor -> IO ()
  set t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_set t0' t1'

  set1d :: Tensor -> CLLong -> HsReal Tensor -> IO ()
  set1d t l0 l1 = withForeignPtr (tensor t) (\t' -> Sig.c_set1d t' l0 l1)

  set2d :: Tensor -> CLLong -> CLLong -> HsReal Tensor -> IO ()
  set2d t l0 l1 l2 = withForeignPtr (tensor t) (\t' -> Sig.c_set2d t' l0 l1 l2)

  set3d :: Tensor -> CLLong -> CLLong -> CLLong -> HsReal Tensor -> IO ()
  set3d t l0 l1 l2 l3 = withForeignPtr (tensor t) (\t' -> Sig.c_set3d t' l0 l1 l2 l3)

  set4d :: Tensor -> CLLong -> CLLong -> CLLong -> CLLong -> HsReal Tensor -> IO ()
  set4d t l0 l1 l2 l3 l4 = withForeignPtr (tensor t) (\t' -> Sig.c_set4d t' l0 l1 l2 l3 l4)

  setFlag :: Tensor -> CChar -> IO ()
  setFlag t l0 = withForeignPtr (tensor t) (\t' -> Sig.c_setFlag t' l0)

  setStorage :: Tensor -> HsStorage Tensor -> CPtrdiff -> Ptr CTHLongStorage -> Ptr CTHLongStorage -> IO ()
  setStorage t s a b c =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage t' s' a b c


  setStorage1d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> IO ()
  setStorage1d t s pd d00 d01 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage1d t' s' pd d00 d01


  setStorage2d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage2d t s pd d00 d01 d10 d11 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage2d t' s' pd d00 d01 d10 d11


  setStorage3d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage3d t s pd d00 d01 d10 d11 d20 d21 =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage3d t' s' pd d00 d01 d10 d11 d20 d21


  setStorage4d :: Tensor -> HsStorage Tensor -> CPtrdiff -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()
  setStorage4d t s pd d00 d01 d10 d11 d20 d21 d30 d31  =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorage4d t' s' pd d00 d01 d10 d11 d20 d21 d30 d31


  setStorageNd :: Tensor -> HsStorage Tensor -> CPtrdiff -> CInt -> Ptr CLLong -> Ptr CLLong -> IO ()
  setStorageNd t s a b c d =
    withForeignPtr (tensor t) $ \t' ->
      withForeignPtr (storage s) $ \s' ->
        Sig.c_setStorageNd t' s' a b c d


  size :: Tensor -> CInt -> IO CLLong
  size t l0 = withForeignPtr (tensor t) (\t' -> pure $ Sig.c_size t' l0)

  sizeDesc :: Tensor -> IO CTHDescBuff
  sizeDesc t = withForeignPtr (tensor t) (pure . Sig.c_sizeDesc)

  squeeze :: Tensor -> Tensor -> IO ()
  squeeze t0 t1 =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze t0' t1'

  squeeze1d :: Tensor -> Tensor -> CInt -> IO ()
  squeeze1d t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_squeeze1d t0' t1' d

  storage :: Tensor -> IO (HsStorage Tensor)
  storage t = withForeignPtr (tensor t) (pure . Sig.c_storage) >>= asStorage

  storageOffset :: Tensor -> IO CPtrdiff
  storageOffset t = withForeignPtr (tensor t) (pure . Sig.c_storageOffset)

  stride :: Tensor -> CInt -> IO CLLong
  stride t a = withForeignPtr (tensor t) (\t' -> pure $ Sig.c_stride t' a)

  transpose :: Tensor -> Tensor -> CInt -> CInt -> IO ()
  transpose t0 t1 a b =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_transpose t0' t1' a b

  unfold :: Tensor -> Tensor -> CInt -> CLLong -> CLLong -> IO ()
  unfold t0 t1 a b c =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unfold t0' t1' a b c

  unsqueeze1d :: Tensor -> Tensor -> CInt -> IO ()
  unsqueeze1d t0 t1 d =
    withForeignPtr (tensor t0) $ \t0' ->
      withForeignPtr (tensor t1) $ \t1' ->
        Sig.c_unsqueeze1d t0' t1' d


{-
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


class IsList Tensor => DynamicTH Tensor where
  printTensor :: Tensor -> IO ()
  fromListNd :: SomeDims -> [Item t] -> t
  fromList1d :: [Item t] -> t
  resize :: Tensor -> SomeDims -> t
  get :: SomeDims -> Tensor -> Item t
  newWithTensor :: Tensor -> t
  new :: SomeDims -> t
  new_ :: SomeDims -> IO Tensor
  free_ :: Tensor -> IO ()
  init :: SomeDims -> Item Tensor -> t
  transpose :: Word -> Word -> Tensor -> t
  trans :: Tensor -> t
  shape :: Tensor -> SomeDims


instance DynamicTH TensorDouble where
  printTensor = td_p
  fromListNd (SomeDims d) = td_fromListNd d
  fromList1d = td_fromList1d
  resize Tensor (SomeDims d) = td_resize Tensor d
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
td_p Tensor = withForeignPtr (getForeign t) Gen.dispRawRealFloat

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
  mutTensor Tensor (idx, value) = withForeignPtr Tensor $ \tp ->
    Gen.c_set1d tp (fromIntegral idx) (realToFrac value)
{-# NOINLINE td_fromList1d #-}

-- |Copy contents of tensor into a new one of specified size
td_resize :: k ~ Nat => TensorDouble -> Dim (d::[k]) -> TensorDouble
td_resize Tensor d = unsafePerformIO $ do
  let resDummy = td_new d
  newPtr <- withForeignPtr (getForeign t) Gen.c_newClone
  newFPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  withForeignPtr newFPtr (withForeignPtr (getForeign resDummy) . Gen.c_resizeAs)
  pure $ TensorDouble newFPtr
{-# NOINLINE td_resize #-}

td_get :: Dim (d::[k]) -> TensorDouble -> Double
td_get loc tensor = unsafePerformIO $ withForeignPtr
  (getForeign tensor)
  (\t -> pure . realToFrac $ Tensor `Gen.genericGet` loc)
{-# NOINLINE td_get #-}

td_newWithTensor :: TensorDouble -> TensorDouble
td_newWithTensor Tensor = unsafePerformIO $ do
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
td_free_ Tensor = finalizeForeignPtr $! getForeign t

td_init :: k ~ Nat => Dim (d::[k]) -> Double -> TensorDouble
td_init ds val = unsafePerformIO $ do
  newPtr <- Gen.constant ds (realToFrac val)
  fPtr <- newForeignPtr T.p_THDoubleTensor_free newPtr
  withForeignPtr fPtr (Gen.inplaceFill realToFrac val)
  pure $ TensorDouble fPtr -- (SomeDims ds)
{-# NOINLINE td_init #-}

td_transpose :: Word -> Word -> TensorDouble -> TensorDouble
td_transpose dim1 dim2 Tensor = unsafePerformIO $ do
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
td_trans Tensor = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p 1 0)
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ TensorDouble newFPtr
{-# NOINLINE td_trans #-}

td_shape :: TensorDouble -> SomeDims
td_shape Tensor = unsafePerformIO $ withForeignPtr (getForeign t) (pure . Gen.getDynamicDim)
{-# NOINLINE td_shape #-}

shapeList :: DynamicTH Tensor => Tensor -> [Int]
shapeList = Dim.dimVals' . shape

rank :: DynamicTH Tensor => Tensor -> Int
rank = Dim.rank' . shape
-}
