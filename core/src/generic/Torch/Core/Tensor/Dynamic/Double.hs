{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module Torch.Core.Tensor.Dynamic.Double (
  disp,
  td_p,
  td_new,
  td_init,
  -- TODO - use this convention for everything
  td_get,
  td_newWithTensor,
  td_transpose,
  td_trans,
  )
where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Internal (w2cl, i2cl)
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import THDoubleLapack

disp tensor =
  (withForeignPtr(tdTensor tensor) dispRaw)

td_p = disp


td_get :: TensorDim Integer -> TensorDouble -> IO Double
td_get loc tensor =
  withForeignPtr
    (tdTensor tensor)
    (\t -> pure . realToFrac $ getter loc t)
  where
    getter D0 t = undefined
    getter (D1 d1) t = c_THDoubleTensor_get1d t $ i2cl d1
    getter (D2 (d1, d2)) t = c_THDoubleTensor_get2d t
                             (i2cl d1) (i2cl d2)
    getter (D3 (d1, d2, d3)) t = c_THDoubleTensor_get3d t
                                 (i2cl d1) (i2cl d2) (i2cl d3)
    getter (D4 (d1, d2, d3, d4)) t = c_THDoubleTensor_get4d t
                                     (i2cl d1) (i2cl d2) (i2cl d3) (i2cl d4)

td_newWithTensor :: TensorDouble -> TensorDouble
td_newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (
    \tPtr -> c_THDoubleTensor_newWithTensor tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)

-- |Create a new (double) tensor of specified dimensions and fill it with 0
td_new :: TensorDim Word -> TensorDouble
td_new dims = unsafePerformIO $ do
  newPtr <- tensorRaw dims 0.0
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble fPtr dims

td_init :: TensorDim Word -> Double -> TensorDouble
td_init dims value = unsafePerformIO $ do
  newPtr <- tensorRaw dims value
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr (fillRaw value)
  pure $ TensorDouble fPtr dims

td_transpose :: Word -> Word -> TensorDouble -> TensorDouble
td_transpose dim1 dim2 t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (
    \tPtr -> c_THDoubleTensor_newTranspose tPtr dim1C dim2C
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)
  where
    dim1C = fromIntegral dim1
    dim2C = fromIntegral dim2

td_trans :: TensorDouble -> TensorDouble
td_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (
    \tPtr -> c_THDoubleTensor_newTranspose tPtr 1 0
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble newFPtr (dimFromRaw newPtr)

test :: IO ()
test = do
  let foo = td_new (D1 5)
  -- disp foo
  let t = td_init (D2 (5, 2)) 3.0
  disp $ td_transpose 1 0 (td_transpose 1 0 t)
  pure ()
