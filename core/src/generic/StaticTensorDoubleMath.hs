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

  , tds_cadd
  , tds_csub
  , tds_cmul
  , tds_cpow
  , tds_cdiv
  , tds_clshift
  , tds_crshift
  , tds_cfmod
  , tds_cremainder
  , tds_cbitand
  , tds_cbitor
  , tds_cbitxor
  , tds_addcmul
  , tds_addcdiv
  , tds_addmv

  ) where



import Data.Singletons
-- import Data.Singletons.Prelude
import Data.Singletons.TypeLits
import Foreign (Ptr)
import Foreign.C.Types (CLong, CDouble)
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

-- ----------------------------------------
-- c* cadd, cmul, cdiv, cpow, ...
-- ----------------------------------------

-- usually this is 1 mutation arg + 2 parameter args
type Raw3Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- usually this is 1 mutation arg + 3 parameter args
type Raw4Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor ->Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

apply2 :: SingI d => Raw3Arg -> (TDS d) -> (TDS d) -> IO (TDS d)
apply2 fun t src = do
  let r_ = tds_new
  withForeignPtr (tdsTensor r_)
    (\rPtr ->
       withForeignPtr (tdsTensor t)
         (\tPtr ->
            withForeignPtr (tdsTensor src)
              (\srcPtr ->
                  fun rPtr tPtr srcPtr
              )
         )
    )
  pure r_

apply3 :: SingI d => Raw4Arg -> (TDS d) -> (TDS d) -> (TDS d) -> IO (TDS d)
apply3 fun t src1 src2 = do
  let r_ = tds_new
  withForeignPtr (tdsTensor r_)
    (\rPtr ->
       withForeignPtr (tdsTensor t)
         (\tPtr ->
            withForeignPtr (tdsTensor src1)
              (\src1Ptr ->
                 withForeignPtr (tdsTensor src2)
                   (\src2Ptr ->
                      fun rPtr tPtr src1Ptr src2Ptr
                   )
              )
         )
    )
  pure r_



-- argument rotations - used so that the constants are curried and only tensor
-- pointers are needed for apply* functions
-- mnemonic for reasoning about the type signature - "where did the argument end up" defines the new ordering
swap1
  :: (t1 -> t2 -> t3 -> t4 -> t5) -> t3 -> t1 -> t2 -> t4 -> t5
swap1 fun a b c d = fun b c a d
-- a is applied at position 3, type 3 is arg1
-- b is applied at position 1, type 1 is arg2
-- c is applied at position 2, type 2 is arg3
-- d is applied at position 4, type 4 is arg 4

swap2 fun a b c d e = fun b c a d e

swap3 fun a b c d e f = fun c a d b e f

-- cadd = z <- y + scalar * x, z value discarded
-- allocate r_ for the user instead of taking it as an argument
tds_cadd :: SingI d => (TDS d) -> Double -> (TDS d) -> (TDS d)
tds_cadd t scale src = unsafePerformIO $
  apply2 ((swap1 c_THDoubleTensor_cadd) scaleC) t src
  where scaleC = realToFrac scale

tds_csub :: SingI d => (TDS d) -> Double -> (TDS d) -> (TDS d)
tds_csub t scale src = unsafePerformIO $ do
  apply2 ((swap1 c_THDoubleTensor_csub) scaleC) t src
  where scaleC = realToFrac scale

tds_cmul :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cmul t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cmul t src

tds_cpow :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cpow t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cpow t src

tds_cdiv :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cdiv t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cdiv t src

tds_clshift :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_clshift t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_clshift t src

tds_crshift :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_crshift t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_crshift t src

tds_cfmod :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cfmod t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cfmod t src

tds_cremainder :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cremainder t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cremainder t src

tds_cbitand :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cbitand  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitand t src

tds_cbitor :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cbitor  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitor t src

tds_cbitxor :: SingI d => (TDS d) -> (TDS d) -> (TDS d)
tds_cbitxor t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitxor t src

tds_addcmul :: SingI d => (TDS d) -> Double -> (TDS d) -> (TDS d) -> (TDS d)
tds_addcmul t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcmul) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

tds_addcdiv :: SingI d => (TDS d) -> Double -> (TDS d) -> (TDS d) -> (TDS d)
tds_addcdiv t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcdiv) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

tds_addmv :: SingI d => Double -> (TDS d) -> Double -> (TDS d) -> (TDS d) -> (TDS d)
tds_addmv beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

