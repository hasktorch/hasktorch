{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Core.Tensor.Dynamic.Double
  (td_p
  , td_new
  , td_new_
  , td_fromList
  , td_init
  , td_free_
  , td_get
  , td_newWithTensor
  , td_resize
  , td_transpose
  , td_trans
  ) where

import Foreign (finalizeForeignPtr)
import Foreign.C.Types
import Foreign.ForeignPtr (withForeignPtr, newForeignPtr)
import GHC.Exts (fromList, toList, IsList, Item, Ptr)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (i2cll, onDims, impossible)
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import THTypes
import THDoubleTensor

td_p :: TensorDouble -> IO ()
td_p tensor = withForeignPtr (tdTensor tensor) dispRaw

instance IsList (TensorDouble) where
  type Item (TensorDouble) = Double
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
td_fromList :: [Double] -> TensorDim Word -> TensorDouble
td_fromList l d = case fromIntegral (product d) == length l of
  True -> td_resize (td_fromList1D l) d
  False -> error "Incorrect tensor dimensions specified."

-- |Copy contents of tensor into a new one of specified size
td_resize :: TensorDouble -> TensorDim Word -> TensorDouble
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

td_get :: TensorDim Integer -> TensorDouble -> IO Double
td_get loc tensor =
  withForeignPtr
    (tdTensor tensor)
    (\t -> pure . realToFrac . getter loc $ t)
  where
    getter :: TensorDim Integer -> Ptr CTHDoubleTensor -> CDouble
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
td_new :: TensorDim Word -> TensorDouble
td_new dims = unsafePerformIO $ do
  newPtr <- tensorRaw dims 0.0
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr dims
{-# NOINLINE td_new #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new_ :: TensorDim Word -> IO TensorDouble
td_new_ dims = do
  newPtr <- tensorRaw dims 0.0
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr dims

td_free_ :: TensorDouble -> IO ()
td_free_ t =
  finalizeForeignPtr $! (tdTensor t)

td_init :: TensorDim Word -> Double -> TensorDouble
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
