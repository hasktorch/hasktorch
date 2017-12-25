{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}


module Torch.Core.Tensor.Generic where

-- import Torch.Core.Tensor.Double as D
import Numeric.Dimensions (Dim(..))
import Foreign (Ptr)
import Foreign (ForeignPtr, finalizeForeignPtr)
import Foreign.C.Types
import Foreign.ForeignPtr (withForeignPtr, newForeignPtr)
import GHC.Exts (fromList, toList, IsList, Item, Ptr)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (i2cll, onDims, impossible)
import Torch.Core.Tensor.Dim
-- import Torch.Core.Tensor.Raw
-- import Torch.Core.Tensor.Types
import THTypes
import qualified THDoubleTensor as T
import qualified THDoubleTensorRandom as R (c_THDoubleTensor_uniform)
import qualified THDoubleTensorMath as M (c_THDoubleTensor_sigmoid, c_THDoubleTensor_fill)
import qualified THRandom as R (c_THGenerator_new)

class GenericRawTensor t where
  type CType t
  type HaskType t
  data HaskTensor t

  -- | flatten a TH-tensor into a list, given a pointer to the TH-tensor
  flatten :: Ptr t -> [CType t]

  -- | randomly initialize a tensor with uniform random values from a range
  randInit :: Ptr CTHGenerator -> Dim (d::[k]) -> CType t {-lower-} -> CType t {-upper-} -> IO (Ptr t)

  -- -- | Create a new tensor of specified dimensions and fill it with 0
  constant :: Dim (d :: [k]) -> HaskType t -> IO (Ptr t)

  -- | make rank-0 tensor
  zeros0d :: Dim (d :: [k]) -> IO (HaskTensor t)
  zeros1d :: Dim (d :: [k]) -> IO (HaskTensor t)
  zeros2d :: Dim (d :: [k]) -> IO (HaskTensor t)
  zeros3d :: Dim (d :: [k]) -> IO (HaskTensor t)
  zeros4d :: Dim (d :: [k]) -> IO (HaskTensor t)

  inplace :: (Ptr t -> Ptr t -> IO ()) -> Ptr t -> IO (Ptr t)
  fill :: HaskType t -> Ptr t -> IO ()
  dimList :: Ptr t -> [Int]
  dimView :: Ptr t -> DimView

type TensorDouble = HaskTensor CTHDoubleTensor


instance GenericRawTensor CTHDoubleTensor where
  type CType CTHDoubleTensor = CDouble
  type HaskType CTHDoubleTensor = Double
  newtype HaskTensor CTHDoubleTensor = THDouble { ctensor :: ForeignPtr CTHDoubleTensor }

  -- | flatten a CTHDoubleTensor into a list
  flatten :: Ptr CTHDoubleTensor -> [CDouble]
  flatten tensor =
    case map getDim [0 .. T.c_THDoubleTensor_nDimension tensor - 1] of
      []           -> mempty
      [x]          -> T.c_THDoubleTensor_get1d tensor <$> range x
      [x, y]       -> T.c_THDoubleTensor_get2d tensor <$> range x <*> range y
      [x, y, z]    -> T.c_THDoubleTensor_get3d tensor <$> range x <*> range y <*> range z
      [x, y, z, q] -> T.c_THDoubleTensor_get4d tensor <$> range x <*> range y <*> range z <*> range q
      _ -> error "TH doesn't support getting tensors higher than 4-dimensions"
    where
      getDim :: CInt -> Int
      getDim = fromIntegral . T.c_THDoubleTensor_size tensor

      range :: Integral i => Int -> [i]
      range mx = [0 .. fromIntegral mx - 1]

  -- |randomly initialize a tensor with uniform random values from a range
  -- TODO - finish implementation to handle sizes correctly
  randInit
    :: Ptr CTHGenerator
    -> Dim (dims :: [k])
    -> CDouble
    -> CDouble
    -> IO (Ptr CTHDoubleTensor)
  randInit gen dims lower upper = do
    t <- constant dims 0.0
    R.c_THDoubleTensor_uniform t gen lower upper
    pure t

  zeros0d = undefined
  zeros1d = undefined
  zeros2d = undefined
  zeros3d = undefined
  zeros4d = undefined

  -- | Returns a function that accepts a tensor and fills it with specified value
  -- and returns the IO context with the mutated tensor
  fill :: Double -> Ptr CTHDoubleTensor -> IO ()
  fill = flip M.c_THDoubleTensor_fill . realToFrac

  -- | Create a new (double) tensor of specified dimensions and fill it with 0
  -- safe version
  constant :: Dim (ns::[k]) -> Double -> IO (Ptr CTHDoubleTensor)
  constant dims value = do
    newPtr <- go dims
    fill value newPtr
    pure newPtr
    where
      go :: Dim (ns::[k]) -> IO (Ptr CTHDoubleTensor)
      go = onDims fromIntegral
        T.c_THDoubleTensor_new
        T.c_THDoubleTensor_newWithSize1d
        T.c_THDoubleTensor_newWithSize2d
        T.c_THDoubleTensor_newWithSize3d
        T.c_THDoubleTensor_newWithSize4d

  -- |apply a tensor transforming function to a tensor
  inplace
    :: (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())
    -> Ptr CTHDoubleTensor
    -> IO (Ptr CTHDoubleTensor)
  inplace f t1 = do
    r_ <- T.c_THDoubleTensor_new
    f r_ t1
    pure r_

  -- |Dimensions of a raw tensor as a list
  dimList :: Ptr CTHDoubleTensor -> [Int]
  dimList t = getDim <$> [0 .. T.c_THDoubleTensor_nDimension t - 1]
    where
      getDim :: CInt -> Int
      getDim = fromIntegral . T.c_THDoubleTensor_size t

  -- |Dimensions of a raw tensor as a TensorDim value
  dimView :: Ptr CTHDoubleTensor -> DimView
  dimView raw =
    case length sz of
      0 -> D0
      1 -> D1 (at 0)
      2 -> D2 (at 0) (at 1)
      3 -> D3 (at 0) (at 1) (at 2)
      4 -> D4 (at 0) (at 1) (at 2) (at 3)
      5 -> D5 (at 0) (at 1) (at 2) (at 3) (at 5)
      _ -> undefined -- TODO - make this safe
    where
      sz :: [Int]
      sz = dimList raw

      at :: Int -> Int
      at n = fromIntegral (sz !! n)
-- -- |Fill a raw Double tensor with 0.0
-- fillRaw0 :: TensorDoubleRaw -> IO TensorDoubleRaw
-- fillRaw0 tensor = fillRaw 0.0 tensor >> pure tensor

{-
  shape :: Ptr t -> [Int]

  render :: hs -> IO ()
  new :: hs -> IO hs
  new_ :: hs -> IO ()
  init :: ()
  free_ :: hs -> IO ()
  get :: ()
  newWithTensor :: ()
  resize :: ()
  transpose :: ()
  trans :: ()

td_p :: TensorDouble (Dim ds) -> IO ()
td_p tensor = withForeignPtr (tdTensor tensor) dispRaw

instance IsList (TensorDouble (ds::[k])) where
  type Item (TensorDouble (ds::[k])) = Double
  fromList l = unsafePerformIO $ go result
    where
      result = td_new (D1 (fromIntegral $ length l))
      go t = do
        mapM_ mutTensor (zip [0..(length l) - 1] l)
        pure t
        where
          mutTensor (idx, value) =
            let (idxC, valueC) = (fromIntegral idx, realToFrac value) in
              withForeignPtr (tdTensor t)
                (\tp -> do
                    c_THDoubleTensor_set1d tp idxC valueC
                )
  {-# NOINLINE fromList #-}
  toList t = undefined -- TODO

-- |Initialize a 1D tensor from a list
td_fromList1D :: [Double] -> TensorDouble
td_fromList1D l = fromList l

-- |Initialize a tensor of arbitrary dimension from a list
td_fromList :: [Double] -> Dim (ds::[k]) -> TensorDouble
td_fromList l d = case fromIntegral (product d) == length l of
  True -> td_resize (td_fromList1D l) d
  False -> error "Incorrect tensor dimensions specified."

-- |Copy contents of tensor into a new one of specified size
td_resize :: TensorDouble -> Dim (ds::[k]) -> TensorDouble
td_resize t d = unsafePerformIO $ do
  let resDummy = td_new d
  newPtr <- withForeignPtr (tdTensor t) (
    \tPtr ->
      c_THDoubleTensor_newClone tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr (newFPtr)
    (\selfp ->
        withForeignPtr (tdTensor resDummy)
          (\srcp ->
             c_THDoubleTensor_resizeAs selfp srcp
          )
    )
  pure $ TensorDouble newFPtr d
{-# NOINLINE td_resize #-}

td_get :: Dim (ds::[ns]) -> TensorDouble -> IO Double
td_get loc tensor =
  withForeignPtr
    (tdTensor tensor)
    (\t -> pure . realToFrac . getter loc $ t)
  where
    getter :: Dim (ds::[ns]) -> Ptr CTHDoubleTensor -> CDouble
    getter dim t = onDims i2cll
      (impossible "0-rank will never be called")
      (c_THDoubleTensor_get1d t)
      (c_THDoubleTensor_get2d t)
      (c_THDoubleTensor_get3d t)
      (c_THDoubleTensor_get4d t)
      dim

td_newWithTensor :: TensorDouble -> TensorDouble
td_newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (\tPtr -> c_THDoubleTensor_newWithTensor tPtr)
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)
{-# NOINLINE td_newWithTensor #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new :: Dim (ds::[ns]) -> TensorDouble
td_new dims = unsafePerformIO $ do
  newPtr <- tensorRaw dims 0.0
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr dims
{-# NOINLINE td_new #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new_ :: Dim (ds::[ns]) -> IO TensorDouble
td_new_ dims = do
  newPtr <- tensorRaw dims 0.0
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr dims

td_free_ :: TensorDouble -> IO ()
td_free_ t =
  finalizeForeignPtr $! (tdTensor t)

td_init :: Dim (ds::[ns]) -> Double -> TensorDouble
td_init dims value = unsafePerformIO $ do
  newPtr <- tensorRaw dims value
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr (fillRaw value)
  pure $ TensorDouble fPtr dims
{-# NOINLINE td_init #-}

td_transpose :: Word -> Word -> TensorDouble -> TensorDouble
td_transpose dim1 dim2 t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (\tPtr -> c_THDoubleTensor_newTranspose tPtr dim1C dim2C)
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)
  where
    dim1C, dim2C :: CInt
    dim1C = fromIntegral dim1
    dim2C = fromIntegral dim2
{-# NOINLINE td_transpose #-}

td_trans :: TensorDouble -> TensorDouble
td_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr
    (tdTensor t)
    (\tPtr -> c_THDoubleTensor_newTranspose tPtr 1 0)
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)
{-# NOINLINE td_trans #-}
-}
