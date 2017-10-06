{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module TensorDouble (
  get_,
  tensorNew_,

  newWithTensor
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

import TensorRaw
import TensorTypes
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import THDoubleLapack

wrap tensor = TensorDouble_ <$> (newForeignPtr p_THDoubleTensor_free tensor)

w2cl = fromIntegral

get_ loc tensor =
   (withForeignPtr(tdTensor tensor) (\t ->
                                        pure $ getter loc t
                                    ))
  where
    getter D0 t = undefined
    getter (D1 d1) t = c_THDoubleTensor_get1d t $ w2cl d1
    getter (D2 d1 d2) t = c_THDoubleTensor_get2d t
                          (w2cl d1) (w2cl d2)
    getter (D3 d1 d2 d3) t = c_THDoubleTensor_get3d t
                             (w2cl d1) (w2cl d2) (w2cl d3)
    getter (D4 d1 d2 d3 d4) t = c_THDoubleTensor_get4d t
                                (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

newWithTensor :: TensorDouble_ -> TensorDouble_
newWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdTensor t) (
    \tPtr -> c_THDoubleTensor_newWithTensor tPtr
    )
  newFPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  pure $ TensorDouble_ newFPtr (dimFromRaw newPtr)


-- |Create a new (double) tensor of specified dimensions and fill it with 0
tensorNew_ :: TensorDim Word -> TensorDouble_
tensorNew_ dims = unsafePerformIO $ do
  newPtr <- go dims
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble_ fPtr dims
  where
    go D0 = c_THDoubleTensor_new
    go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
    go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                    (w2cl d1) (w2cl d2)
    go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                       (w2cl d1) (w2cl d2) (w2cl d3)
    go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                          (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

test :: IO ()
test = do
  let foo = tensorNew_ (D1 5)
  -- disp_ foo
  pure ()
