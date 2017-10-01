module TensorMath (

  fillCopy_,
  fillMutate_,

  addConst,
  subConst,
  mulConst,
  divConst,

  minAll,
  maxAll,
  medianAll,
  sumAll,
  prodAll,
  meanAll,

  neg,
  absT,
  sigmoid,
  logT,
  lgamma

  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import System.IO.Unsafe (unsafePerformIO)

import Tensor
import TensorRaw
import TensorTypes

import THDoubleTensor
import THDoubleTensor
import THDoubleTensorMath
import THTypes


-- ----------------------------------------
-- Tensor fill operations
-- ----------------------------------------

fillCopy_ :: Real a => a -> TensorDouble_ -> TensorDouble_
fillCopy_ value tensor = unsafePerformIO $
  withForeignPtr(tdTensor nt) (\t -> do
                                  fillRaw value t
                                  pure nt
                              )
  where nt = tensorNew_ (tdDim tensor)

fillMutate_ :: Real a => a -> TensorDouble_ -> IO ()
fillMutate_ value tensor =
  withForeignPtr(tdTensor tensor) (\t -> fillRaw value t)


-- ----------------------------------------
-- Tensor-constant operations to constant operations
-- ----------------------------------------

apply1_ transformation mtx val = unsafePerformIO $ do
  withForeignPtr (tdTensor res)
    (\r_ -> withForeignPtr (tdTensor mtx)
            (\t -> do
                transformation r_ t
                pure r_
            )
    )
  pure res
  where
    res = tensorNew_ (tdDim mtx)

addConst :: TensorDouble_ -> Double -> TensorDouble_
addConst mtx val = apply1_ tAdd mtx val
  where
    tAdd r_ t = c_THDoubleTensor_add r_ t (realToFrac val)

subConst :: TensorDouble_ -> Double -> TensorDouble_
subConst mtx val = apply1_ tSub mtx val
  where
    tSub r_ t = c_THDoubleTensor_sub r_ t (realToFrac val)

mulConst :: TensorDouble_ -> Double -> TensorDouble_
mulConst mtx val = apply1_ tMul mtx val
  where
    tMul r_ t = c_THDoubleTensor_mul r_ t (realToFrac val)

divConst :: TensorDouble_ -> Double -> TensorDouble_
divConst mtx val = apply1_ tDiv mtx val
  where
    tDiv r_ t = c_THDoubleTensor_div r_ t (realToFrac val)

-- ----------------------------------------
-- Collapse to constant operations
-- ----------------------------------------

-- |Generalize non-mutating collapse of a tensor to a constant or another tensor
apply0_ :: (Ptr CTHDoubleTensor -> a) -> TensorDouble_ -> a
apply0_ operation tensor = unsafePerformIO $ do
  withForeignPtr (tdTensor tensor) (\t -> pure $ operation t)

minAll :: TensorDouble_ -> Double
minAll tensor = apply0_ tMinAll tensor
  where
    tMinAll t = realToFrac $ c_THDoubleTensor_minall t

maxAll :: TensorDouble_ -> Double
maxAll tensor = apply0_ tMaxAll tensor
  where
    tMaxAll t = realToFrac $ c_THDoubleTensor_maxall t

medianAll :: TensorDouble_ -> Double
medianAll tensor = apply0_ tMedianAll tensor
  where
    tMedianAll t = realToFrac $ c_THDoubleTensor_medianall t

sumAll :: TensorDouble_ -> Double
sumAll tensor = apply0_ tSumAll tensor
  where
    tSumAll t = realToFrac $ c_THDoubleTensor_sumall t

prodAll :: TensorDouble_ -> Double
prodAll tensor = apply0_ tProdAll tensor
  where
    tProdAll t = realToFrac $ c_THDoubleTensor_prodall t

meanAll :: TensorDouble_ -> Double
meanAll tensor = apply0_ tMeanAll tensor
  where
    tMeanAll t = realToFrac $ c_THDoubleTensor_meanall t

-- ----------------------------------------
-- Tensor to Tensor transformation
-- ----------------------------------------

-- |Wrapper to apply tensor -> tensor non-mutating operation
apply0Tensor op resDim t = unsafePerformIO $ do
  let res = tensorNew_ resDim
  withForeignPtr (tdTensor res) (\r_ -> op r_ t)
  pure res

neg :: TensorDouble_ -> TensorDouble_
neg tensor = apply0_ tNeg tensor
  where
    tNeg t = apply0Tensor c_THDoubleTensor_neg (tdDim tensor) t

absT :: TensorDouble_ -> TensorDouble_
absT tensor = apply0_ tAbs tensor
  where
    tAbs t = apply0Tensor c_THDoubleTensor_abs (tdDim tensor) t

sigmoid :: TensorDouble_ -> TensorDouble_
sigmoid tensor = apply0_ tSigmoid tensor
  where
    tSigmoid t = apply0Tensor c_THDoubleTensor_sigmoid (tdDim tensor) t

logT :: TensorDouble_ -> TensorDouble_
logT tensor = apply0_ tLog tensor
  where
    tLog t = apply0Tensor c_THDoubleTensor_log (tdDim tensor) t

lgamma :: TensorDouble_ -> TensorDouble_
lgamma tensor = apply0_ tLgamma tensor
  where
    tLgamma t = apply0Tensor c_THDoubleTensor_lgamma (tdDim tensor) t

-- ----------------------------------------
-- Matrix-vector
-- ----------------------------------------

-- |tag: unsafe
-- TODO - determine how to deal with resource allocation
(#>) :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
mat #> vec = undefined -- unsafePerformIO $ do
  -- res <- fromJust $ tensorNew_ $ [nrows mat]
  -- c_THDoubleTensor_addmv res 1.0 res 1.0 mat vec
  -- pure res
