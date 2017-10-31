{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}

module StaticTensorDoubleMath (
  tds_fill
  , tds_fill_

  , tds_addConst
  , tds_subConst
  , tds_mulConst
  , tds_divConst

  , tds_dot

  , tds_minAll
  , tds_maxAll
  , tds_medianAll
  , tds_sumAll
  , tds_prodAll
  , tds_meanAll

  , tds_neg
  , tds_abs
  , tds_sigmoid
  , tds_log
  , tds_lgamma

  ) where


import Data.Singletons
-- import Data.Singletons.Prelude
import Data.Singletons.TypeLits
import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import System.IO.Unsafe (unsafePerformIO)

import TensorRaw
import TensorDouble
import TensorTypes
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import StaticTensorDouble

-- |Generalize non-mutating collapse of a tensor to a constant or another tensor
apply0_ :: (Ptr CTHDoubleTensor -> a) -> (TDS d) -> IO a
apply0_ operation tensor = do
  withForeignPtr (tdsTensor tensor) (\t -> pure $ operation t)

apply1_ :: SingI d => (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO a)
     -> (TDS d) -> p -> (TDS d)
apply1_ transformation mtx val = unsafePerformIO $ do
  withForeignPtr (tdsTensor res)
    (\r_ -> withForeignPtr (tdsTensor mtx)
            (\t -> do
                transformation r_ t
                pure r_
            )
    )
  pure res
  where
    res = tds_cloneDim mtx

tds_fill :: (Real a, SingI d) => a -> p -> TensorDoubleStatic d
tds_fill value tensor = unsafePerformIO $
  withForeignPtr(tdsTensor nt) (\t -> do
                                  fillRaw value t
                                  pure nt
                              )
  where nt = tds_new

tds_fill_ :: Real a => a -> (TDS d) -> IO ()
tds_fill_ value tensor =
  withForeignPtr(tdsTensor tensor) (\t -> fillRaw value t)

tds_addConst :: (SingI d, Real p) => TDS d -> p -> TDS d
tds_addConst mtx val = apply1_ tAdd mtx val
  where
    tAdd r_ t = c_THDoubleTensor_add r_ t (realToFrac val)

tds_subConst :: (SingI d, Real p) => TDS d -> p -> TDS d
tds_subConst mtx val = apply1_ tSub mtx val
  where
    tSub r_ t = c_THDoubleTensor_sub r_ t (realToFrac val)

tds_mulConst :: (SingI d, Real p) => TDS d -> p -> TDS d
tds_mulConst mtx val = apply1_ tMul mtx val
  where
    tMul r_ t = c_THDoubleTensor_mul r_ t (realToFrac val)

tds_divConst :: (SingI d, Real p) => TDS d -> p -> TDS d
tds_divConst mtx val = apply1_ tDiv mtx val
  where
    tDiv r_ t = c_THDoubleTensor_div r_ t (realToFrac val)

tds_dot :: (TDS d) -> (TDS d) -> Double
tds_dot t src = realToFrac $ unsafePerformIO $ do
  withForeignPtr (tdsTensor t)
    (\tPtr -> withForeignPtr (tdsTensor src)
      (\srcPtr ->
          pure $ c_THDoubleTensor_dot tPtr srcPtr
      )
    )


tds_minAll :: (TDS d) -> Double
tds_minAll tensor = unsafePerformIO $ apply0_ tMinAll tensor
  where
    tMinAll t = realToFrac $ c_THDoubleTensor_minall t

tds_maxAll :: (TDS d) -> Double
tds_maxAll tensor = unsafePerformIO $ apply0_ tMaxAll tensor
  where
    tMaxAll t = realToFrac $ c_THDoubleTensor_maxall t

tds_medianAll :: (TDS d) -> Double
tds_medianAll tensor = unsafePerformIO $ apply0_ tMedianAll tensor
  where
    tMedianAll t = realToFrac $ c_THDoubleTensor_medianall t

tds_sumAll :: (TDS d) -> Double
tds_sumAll tensor = unsafePerformIO $ apply0_ tSumAll tensor
  where
    tSumAll t = realToFrac $ c_THDoubleTensor_sumall t

tds_prodAll :: (TDS d) -> Double
tds_prodAll tensor = unsafePerformIO $ apply0_ tProdAll tensor
  where
    tProdAll t = realToFrac $ c_THDoubleTensor_prodall t

tds_meanAll :: (TDS d) -> Double
tds_meanAll tensor = unsafePerformIO $ apply0_ tMeanAll tensor
  where
    tMeanAll t = realToFrac $ c_THDoubleTensor_meanall t

-- ----------------------------------------
-- Tensor to Tensor transformation
-- ----------------------------------------

-- |Wrapper to apply tensor -> tensor non-mutating operation
apply0Tensor
  :: SingI d => (Ptr CTHDoubleTensor -> t -> IO a) -> t -> (TDS d)
apply0Tensor op t = unsafePerformIO $ do
  let res = tds_new
  withForeignPtr (tdsTensor res) (\r_ -> op r_ t)
  pure res

tds_neg :: SingI d => TDS d -> TDS d
tds_neg tensor = unsafePerformIO $ apply0_ tNeg tensor
  where
    tNeg t = apply0Tensor c_THDoubleTensor_neg t

tds_abs :: SingI d => TDS d -> TDS d
tds_abs tensor = unsafePerformIO $ apply0_ tAbs tensor
  where
    tAbs t = apply0Tensor c_THDoubleTensor_abs t

tds_sigmoid :: SingI d => TDS d -> TDS d
tds_sigmoid tensor = unsafePerformIO $ apply0_ tSigmoid tensor
  where
    tSigmoid t = apply0Tensor c_THDoubleTensor_sigmoid t

tds_log :: SingI d => TDS d -> TDS d
tds_log tensor = unsafePerformIO $ apply0_ tLog tensor
  where
    tLog t = apply0Tensor c_THDoubleTensor_log t

tds_lgamma :: SingI d => TDS d -> TDS d
tds_lgamma tensor = unsafePerformIO $ apply0_ tLgamma tensor
  where
    tLgamma t = apply0Tensor c_THDoubleTensor_lgamma t

