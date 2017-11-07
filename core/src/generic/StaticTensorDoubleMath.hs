{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}

module StaticTensorDoubleMath (

  (^+^),
  (^-^),
  (!*),
  (^+),
  (^-),
  (+^),
  (-^),
  (^*),
  (^/),
  (*^),
  (/^),
  (<.>),

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
  , tds_cinv
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
  -- , tds_addcmul
  -- , tds_addcdiv
  , tds_addmv
  , tds_mv

  , tds_addmm
  , tds_addr
  , tds_outer
  , tds_addbmm
  , tds_baddbmm
  , tds_match
  , tds_numel
  , tds_max
  , tds_min
  , tds_kthvalue
  , tds_mode
  , tds_median
  , tds_sum
  , tds_prod
  , tds_cumsum
  , tds_cumprod
  , tds_sign
  , tds_trace
  , tds_cross


  ) where

import Data.Singletons
-- import Data.Singletons.Prelude
import Data.Singletons.TypeLits
import Foreign (Ptr)
import Foreign.C.Types (CLong, CDouble, CInt)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import System.IO.Unsafe (unsafePerformIO)

import TensorRaw
import TensorDouble
import TensorLong
import TensorTypes
import THTypes
import THDoubleTensor
import THDoubleTensorMath
import StaticTensorDouble

{- Operators -}

-- |Experimental num instance for static tensors
instance SingI d => Num (TensorDoubleStatic d) where
  (+) t1 t2 = tds_cadd t1 1.0 t2
  (-) t1 t2 = tds_csub t1 1.0  t2
  (*) t1 t2 = tds_cmul t1 t2
  abs t = tds_abs t
  signum t = error "signum not defined for tensors"
  fromInteger t = error "signum not defined for tensors"

(^+^) t1 t2 = tds_cadd t1 1.0 t2
(^-^) t1 t2 = tds_csub t1 1.0 t2

(!*) :: (KnownNat c, KnownNat r) => (TDS '[r, c]) -> (TDS '[c]) -> (TDS '[r])
(!*) m v = tds_mv m v

(^+) :: (Real p, SingI d) => TDS d -> p -> TDS d
(^+) = tds_addConst

(^-) :: (Real p, SingI d) => TDS d -> p -> TDS d
(^-) = tds_subConst

(^*) :: (SingI d, Real p) => TDS d -> p -> TDS d
(^*) = tds_mulConst

(^/) :: (SingI d, Real p) => TDS d -> p -> TDS d
(^/) = tds_divConst

(+^) :: (Real p, SingI d) => p -> TDS d -> TDS d
(+^) = flip tds_addConst

(-^) :: (Real p, SingI d) => p -> TDS d -> TDS d
(-^) val t = tds_addConst (tds_neg t) val

(*^) :: (SingI d, Real p) => p -> TDS d -> TDS d
(*^) = flip tds_mulConst

(/^) :: (SingI d, Real p) => p -> TDS d -> TDS d
(/^) val t = tds_mulConst (tds_cinv t) val

(<.>) t1 t2 = tds_dot t1 t2

{- Helper functions -}

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

tds_cinv :: SingI d => TDS d -> TDS d
tds_cinv tensor = unsafePerformIO $ apply0_ tInv tensor
  where
    tInv t = apply0Tensor c_THDoubleTensor_cinv t

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

apply2 :: (SingI d1, SingI d2, SingI d3) =>
  Raw3Arg -> (TDS d1) -> (TDS d2) -> IO (TDS d3)
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

apply3 :: (SingI d1, SingI d2, SingI d3, SingI d4) =>
  Raw4Arg -> (TDS d1) -> (TDS d2) -> (TDS d3) -> IO (TDS d4)
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

-- TODO - fix constraints on type signatures for addcmul and addcdiv

-- tds_addcmul :: SingI d => (TDS d) -> Double -> (TDS d) -> (TDS d) -> (TDS d)
-- tds_addcmul t scale src1 src2 = unsafePerformIO $ do
--   apply3 ((swap2 c_THDoubleTensor_addcmul) scaleC) t src1 src2
--   where scaleC = (realToFrac scale) :: CDouble

-- tds_addcdiv :: SingI d => (TDS d) -> Double -> (TDS d) -> (TDS d) -> (TDS d)
-- tds_addcdiv t scale src1 src2 = unsafePerformIO $ do
--   apply3 ((swap2 c_THDoubleTensor_addcdiv) scaleC) t src1 src2
--   where scaleC = (realToFrac scale) :: CDouble

-- |beta * t + alpha * (src1 #> src2)
tds_addmv :: (KnownNat c, KnownNat r) =>
  Double -> (TDS '[r]) -> Double -> (TDS '[r, c]) -> (TDS '[c]) -> (TDS '[r])
tds_addmv beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

-- | added simplified use of addmv: src1 #> src2
tds_mv :: (KnownNat c, KnownNat r) => (TDS '[r, c]) -> (TDS '[c]) -> (TDS '[r])
tds_mv m v = tds_addmv 0.0 tds_new 1.0 m v

apply1 fun t = do
  let r_ = tds_new
  withForeignPtr (tdsTensor r_)
    (\rPtr ->
       withForeignPtr (tdsTensor t)
         (\tPtr ->
            fun rPtr tPtr
         )
    )
  pure r_

type Ret2Fun =
  Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

ret2 :: SingI d => Ret2Fun -> (TDS d) -> Int -> Bool -> IO ((TDS d), TensorLong)
ret2 fun t dimension keepdim = do
  let values_ = tds_new
  let indices_ = tl_new (tds_dim t)
  withForeignPtr (tdsTensor values_)
    (\vPtr ->
       withForeignPtr (tlTensor indices_)
         (\iPtr ->
            withForeignPtr (tdsTensor t)
              (\tPtr ->
                  fun vPtr iPtr tPtr dimensionC keepdimC
              )
         )
    )
  pure (values_, indices_)
  where
    keepdimC = if keepdim then 1 else 0
    dimensionC = fromIntegral dimension

-- TODO- add proper type signature with dimensions specified
tds_addmm beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmm) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

tds_addr :: (KnownNat r, KnownNat c) =>
  Double -> TDS '[r, c]-> Double -> TDS '[r] -> TDS '[c]-> TDS '[r, c]
tds_addr beta t alpha vec1 vec2 = unsafePerformIO $ do
  let r_ = tds_new
  withForeignPtr (tdsTensor r_)
    (\rPtr ->
       withForeignPtr (tdsTensor t)
         (\tPtr ->
            withForeignPtr (tdsTensor vec1)
              (\vec1Ptr ->
                 withForeignPtr (tdsTensor vec2)
                   (\vec2Ptr ->
                      c_THDoubleTensor_addr rPtr betaC tPtr alphaC vec1Ptr vec2Ptr
                   )
              )
         )
    )
  pure r_
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

tds_outer :: (KnownNat r, KnownNat c) =>
             TDS '[r] -> TDS '[c] -> TDS '[r, c]
tds_outer vec1 vec2 = tds_addr 0.0 tds_new 1.0 vec1 vec2

-- TODO- add proper type signature with dimensions specified
-- tds_addbmm :: Double -> (TDS d) -> Double -> (TDS d) -> (TDS d) -> (TDS d)
tds_addbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

-- TODO- add proper type signature with dimensions specified
-- tds_baddbmm :: Double -> (TDS d) -> Double -> (TDS d) -> (TDS d) -> (TDS d)
tds_baddbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_baddbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

-- tds_match :: (TDS d) -> (TDS d) -> Double -> (TDS d)
tds_match m1 m2 gain = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_match) gainC) m1 m2
  where
    gainC = realToFrac gain
    swap fun gain b c d = fun b c d gain

tds_numel :: (TDS d) -> Int
tds_numel t = unsafePerformIO $ do
  result <- apply0_ c_THDoubleTensor_numel t
  pure $ fromIntegral result

{-
TODO : need type computations for resulting dimensions
-}

-- TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tds_max :: SingI d => (TDS d) -> Int -> Bool -> ((TDS d), TensorLong)
tds_max t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_max t dimension keepdim

-- TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
-- tds_minT :: (TDS d) -> Int -> Bool -> ((TDS d), TensorLong)
tds_min t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_min t dimension keepdim

-- TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
-- tds_kthvalue :: (TDS d) -> Int -> Int -> Bool -> ((TDS d), TensorLong)
tds_kthvalue t k dimension keepdim = unsafePerformIO $
  ret2 ((swap c_THDoubleTensor_kthvalue) kC) t dimension keepdim
  where
    swap fun a b c d e f = fun b c d a e f -- curry k (4th argument)
    kC = fromIntegral k

-- TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
-- tds_mode :: (TDS d) -> Int -> Bool -> ((TDS d), TensorLong)
tds_mode t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_mode t dimension keepdim

-- TH_API void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tds_median :: SingI d => (TDS d) -> Int -> Bool -> ((TDS d), TensorLong)
tds_median t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_median t dimension keepdim

-- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
-- tds_sum :: (TDS d) -> Int -> Bool -> (TDS d)
tds_sum t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_sum) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
-- tds_prod :: (TDS d) -> Int -> Bool -> (TDS d)
tds_prod t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_prod) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
-- tds_cumsum :: (TDS d) -> Int -> (TDS d)
tds_cumsum t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumsum) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
-- tds_cumprod :: (TDS d) -> Int -> (TDS d)
tds_cumprod t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumprod) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);
tds_sign :: SingI d => (TDS d) -> (TDS d)
tds_sign t = unsafePerformIO $ do
  apply1 c_THDoubleTensor_sign t

-- TH_API accreal THTensor_(trace)(THTensor *t);
tds_trace :: SingI d => (TDS d) -> Double
tds_trace t = realToFrac $ unsafePerformIO $ do
  apply0_ c_THDoubleTensor_trace t

-- TH_API void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
-- tds_cross :: (TDS d) -> (TDS d) -> Int -> (TDS d)
tds_cross a b dimension = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_cross) dimensionC) a b
  where
    dimensionC = fromIntegral dimension
    swap fun a b c d = fun b c d a

-- TH_API void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
-- tds_cmax :: (TDS d) -> (TDS d) -> (TDS d)
tds_cmax t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmax t src

-- TH_API void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
-- tds_cmin :: (TDS d) -> (TDS d) -> (TDS d)
tds_cmin t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmin t src


test = do
  print("initialization")
  let t = (tds_init 3.0 :: TDS '[3,2])
  dispS t
  print("trace ")
  print $ tds_trace t
  print("num el")
  print $ tds_numel t
  print("Dot product")
  print ((tds_init 2.0 :: TDS '[5]) <.> (tds_init 2.0 :: TDS '[5]))

  -- .33..
  dispS $ tds_cinv (tds_init 3.0 :: TDS '[5])

  -- 1.25
  dispS $ 5.0 /^ (tds_init 4.0 :: TDS '[10])
  pure ()

  dispS (tds_outer (tds_init 2.0) (tds_init 3.0) :: TDS '[3,2])
