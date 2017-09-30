{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ForeignFunctionInterface#-}

module Tensor (
  fillCopy_,
  fillMutate_,
  get_,
  tensorNew_
  )
where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)
import Numeric (showGFloat)
import System.IO.Unsafe (unsafePerformIO)

import TensorRaw
import TensorTypes
import TensorUtils
import THTypes
import THDoubleTensor

-- test0 = do
--   let tensor = tensorNew_ (D2 8 4)
--   pure ()

-- |basic test of garbage collected tensor
testGCTensor = do
  let t0 = tensorNew_ (D2 8 4)
      t1 = t0
  fillMutate_ 3.0 t1
  t2 <- fillCopy_ 6.0 t1
  disp_ t0
  disp_ t1
  disp_ t2

fillCopy_ value tensor =
  withForeignPtr(tdTensor nt) (\t -> do
                                  fillRaw value t
                                  pure nt
                              )
  where nt = tensorNew_ (tdDim tensor)

fillMutate_ value tensor =
  (withForeignPtr(tdTensor tensor) (\t -> do
                                       fillRaw value t
                                       -- wrap t
                                       pure ()
                                   ))

disp_ tensor =
  (withForeignPtr(tdTensor tensor) disp)

wrap tensor = TensorDouble_ <$> (newForeignPtr p_THDoubleTensor_free tensor)

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


-- |Create a new (double) tensor of specified dimensions and fill it with 0
tensorNew_ :: TensorDim Word -> TensorDouble_
tensorNew_ dims = unsafePerformIO $ do
  newPtr <- go dims
  fPtr <- newForeignPtr p_THDoubleTensor_free newPtr
  withForeignPtr fPtr fillRaw0
  pure $ TensorDouble_ fPtr dims
  where
    wrap ptr = newForeignPtr p_THDoubleTensor_free ptr
    go D0 = c_THDoubleTensor_new
    go (D1 d1) = c_THDoubleTensor_newWithSize1d $ w2cl d1
    go (D2 d1 d2) = c_THDoubleTensor_newWithSize2d
                    (w2cl d1) (w2cl d2)
    go (D3 d1 d2 d3) = c_THDoubleTensor_newWithSize3d
                       (w2cl d1) (w2cl d2) (w2cl d3)
    go (D4 d1 d2 d3 d4) = c_THDoubleTensor_newWithSize4d
                          (w2cl d1) (w2cl d2) (w2cl d3) (w2cl d4)

-- -- |randomly initialize a tensor with uniform random values from a range
-- -- TODO - finish implementation to handle sizes correctly
-- randInit sz lower upper = do
--   gen <- c_THGenerator_new
--   t <- fromJust $ tensorNew sz
--   mapM_ (\x -> do
--             c_THDoubleTensor_uniform t gen lower upper
--             disp t
--         ) [0..3]

