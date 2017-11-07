{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module TensorDoubleMath (

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

  td_fill,
  td_fill_,

  td_addConst,
  td_subConst,
  td_mulConst,
  td_divConst,

  td_dot,

  td_minAll,
  td_maxAll,
  td_medianAll,
  td_sumAll,
  td_prodAll,
  td_meanAll,

  td_neg,
  td_cinv,
  td_abs,
  td_sigmoid,
  td_log,
  td_lgamma,

  td_cadd,
  td_csub,
  td_cmul,
  td_cpow,
  td_cdiv,
  td_clshift,
  td_crshift,
  td_cfmod,
  td_cremainder,
  td_cbitand,
  td_cbitor,
  td_cbitxor,
  td_addcmul,
  td_addcdiv,
  td_addmv,
  td_addmv_fast,
  td_mv,
  td_mv_fast,

  td_addmm,
  td_addbmm,
  td_baddbmm,
  td_match,
  td_numel,
  td_max,
  td_min,
  td_kthvalue,
  td_mode,
  td_median,
  td_sum,
  td_prod,
  td_cumsum,
  td_cumprod,
  td_sign,
  td_trace,
  td_cross,

  td_eye,

  td_equal
  ) where

import Control.Exception
import Control.Monad (unless)

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Lens.Micro
import System.IO.Unsafe (unsafePerformIO)

import TensorDouble
import TensorLong
import TensorRaw
import TensorTypes

import THDoubleTensor
import THDoubleTensor
import THDoubleTensorMath
import THTypes

-- |Experimental num instance for static tensors
instance Num TensorDouble where
  (+) t1 t2 = td_cadd t1 1.0 t2
  (-) t1 t2 = td_csub t1 1.0  t2
  (*) t1 t2 = td_cmul t1 t2
  abs t = td_abs t
  signum t = error "signum not defined for tensors"
  fromInteger t = error "signum not defined for tensors"

(^+^) t1 t2 = td_cadd t1 1.0 t2
(^-^) t1 t2 = td_csub t1 1.0 t2
(!*) = td_mv
(^+) = td_addConst
(^-) = td_subConst
(^*) = td_mulConst
(^/) = td_divConst
(+^) = flip td_addConst
(-^) val t = td_addConst (td_neg t) val
(*^) = flip td_mulConst
(/^) val t = td_mulConst (td_cinv t) val
(<.>) = td_dot

-- ----------------------------------------
-- Foreign pointer application helper functions
-- ----------------------------------------

apply1_
  :: (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO a)
     -> TensorDouble -> p -> TensorDouble
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
    res = td_new (tdDim mtx)

-- |Generalize non-mutating collapse of a tensor to a constant or another tensor
apply0_ :: (Ptr CTHDoubleTensor -> a) -> TensorDouble -> IO a
apply0_ operation tensor = do
  withForeignPtr (tdTensor tensor) (\t -> pure $ operation t)

-- |Wrapper to apply tensor -> tensor non-mutating operation
apply0Tensor :: (Ptr CTHDoubleTensor -> t -> IO a) -> TensorDim Word -> t
  -> TensorDouble
apply0Tensor op resDim t = unsafePerformIO $ do
  let res = td_new resDim
  withForeignPtr (tdTensor res) (\r_ -> op r_ t)
  pure res

-- usually this is 1 mutation arg + 2 parameter args
type Raw3Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- usually this is 1 mutation arg + 3 parameter args
type Raw4Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor ->Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

apply2 :: Raw3Arg -> TensorDouble -> TensorDouble -> IO TensorDouble
apply2 fun t src = do
  let r_ = td_new (tdDim t)
  withForeignPtr (tdTensor r_)
    (\rPtr ->
       withForeignPtr (tdTensor t)
         (\tPtr ->
            withForeignPtr (tdTensor src)
              (\srcPtr ->
                  fun rPtr tPtr srcPtr
              )
         )
    )
  pure r_

apply3 :: Raw4Arg -> TensorDouble -> TensorDouble -> TensorDouble -> IO TensorDouble
apply3 fun t src1 src2 = do
  let r_ = td_new (tdDim t)
  withForeignPtr (tdTensor r_)
    (\rPtr ->
       withForeignPtr (tdTensor t)
         (\tPtr ->
            withForeignPtr (tdTensor src1)
              (\src1Ptr ->
                 withForeignPtr (tdTensor src2)
                   (\src2Ptr ->
                      fun rPtr tPtr src1Ptr src2Ptr
                   )
              )
         )
    )
  pure r_

type Ret2Fun =
  Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

ret2 :: Ret2Fun -> TensorDouble -> Int -> Bool -> IO (TensorDouble, TensorLong)
ret2 fun t dimension keepdim = do
  let values_ = td_new (tdDim t)
  let indices_ = tl_new (tdDim t)
  withForeignPtr (tdTensor values_)
    (\vPtr ->
       withForeignPtr (tlTensor indices_)
         (\iPtr ->
            withForeignPtr (tdTensor t)
              (\tPtr ->
                  fun vPtr iPtr tPtr dimensionC keepdimC
              )
         )
    )
  pure (values_, indices_)
  where
    keepdimC = if keepdim then 1 else 0
    dimensionC = fromIntegral dimension

apply1 fun t = do
  let r_ = td_new (tdDim t)
  withForeignPtr (tdTensor r_)
    (\rPtr ->
       withForeignPtr (tdTensor t)
         (\tPtr ->
            fun rPtr tPtr
         )
    )
  pure r_

-- ----------------------------------------
-- Tensor fill operations
-- ----------------------------------------

td_fill :: Real a => a -> TensorDouble -> TensorDouble
td_fill value tensor = unsafePerformIO $
  withForeignPtr (tdTensor nt) (\t -> do
                                  fillRaw value t
                                  pure nt
                              )
  where nt = td_new (tdDim tensor)

td_fill_ :: Real a => a -> TensorDouble -> IO ()
td_fill_ value tensor =
  withForeignPtr (tdTensor tensor) (\t -> fillRaw value t)

-- ----------------------------------------
-- Tensor-constant operations to constant operations
-- ----------------------------------------

td_addConst :: TensorDouble -> Double -> TensorDouble
td_addConst mtx val = apply1_ tAdd mtx val
  where
    tAdd r_ t = c_THDoubleTensor_add r_ t (realToFrac val)

td_subConst :: TensorDouble -> Double -> TensorDouble
td_subConst mtx val = apply1_ tSub mtx val
  where
    tSub r_ t = c_THDoubleTensor_sub r_ t (realToFrac val)

td_mulConst :: TensorDouble -> Double -> TensorDouble
td_mulConst mtx val = apply1_ tMul mtx val
  where
    tMul r_ t = c_THDoubleTensor_mul r_ t (realToFrac val)

td_divConst :: TensorDouble -> Double -> TensorDouble
td_divConst mtx val = apply1_ tDiv mtx val
  where
    tDiv r_ t = c_THDoubleTensor_div r_ t (realToFrac val)

-- ----------------------------------------
-- Linear algebra
-- ----------------------------------------

td_dot :: TensorDouble -> TensorDouble -> Double
td_dot t src = realToFrac $ unsafePerformIO $ do
  withForeignPtr (tdTensor t)
    (\tPtr -> withForeignPtr (tdTensor src)
      (\srcPtr ->
          pure $ c_THDoubleTensor_dot tPtr srcPtr
      )
    )

-- ----------------------------------------
-- Collapse to constant operations
-- ----------------------------------------

td_minAll :: TensorDouble -> Double
td_minAll tensor = unsafePerformIO $ apply0_ tMinAll tensor
  where
    tMinAll t = realToFrac $ c_THDoubleTensor_minall t

td_maxAll :: TensorDouble -> Double
td_maxAll tensor = unsafePerformIO $ apply0_ tMaxAll tensor
  where
    tMaxAll t = realToFrac $ c_THDoubleTensor_maxall t

td_medianAll :: TensorDouble -> Double
td_medianAll tensor = unsafePerformIO $ apply0_ tMedianAll tensor
  where
    tMedianAll t = realToFrac $ c_THDoubleTensor_medianall t

td_sumAll :: TensorDouble -> Double
td_sumAll tensor = unsafePerformIO $ apply0_ tSumAll tensor
  where
    tSumAll t = realToFrac $ c_THDoubleTensor_sumall t

td_prodAll :: TensorDouble -> Double
td_prodAll tensor = unsafePerformIO $ apply0_ tProdAll tensor
  where
    tProdAll t = realToFrac $ c_THDoubleTensor_prodall t

td_meanAll :: TensorDouble -> Double
td_meanAll tensor = unsafePerformIO $ apply0_ tMeanAll tensor
  where
    tMeanAll t = realToFrac $ c_THDoubleTensor_meanall t

-- ----------------------------------------
-- Tensor to Tensor transformation
-- ----------------------------------------

td_neg :: TensorDouble -> TensorDouble
td_neg tensor = unsafePerformIO $ apply0_ tNeg tensor
  where
    tNeg t = apply0Tensor c_THDoubleTensor_neg (tdDim tensor) t

td_cinv :: TensorDouble -> TensorDouble
td_cinv tensor = unsafePerformIO $ apply0_ cinv tensor
  where
    cinv t = apply0Tensor c_THDoubleTensor_cinv (tdDim tensor) t

td_abs :: TensorDouble -> TensorDouble
td_abs tensor = unsafePerformIO $ apply0_ tAbs tensor
  where
    tAbs t = apply0Tensor c_THDoubleTensor_abs (tdDim tensor) t

td_sigmoid :: TensorDouble -> TensorDouble
td_sigmoid tensor = unsafePerformIO $ apply0_ tSigmoid tensor
  where
    tSigmoid t = apply0Tensor c_THDoubleTensor_sigmoid (tdDim tensor) t

td_log :: TensorDouble -> TensorDouble
td_log tensor = unsafePerformIO $ apply0_ tLog tensor
  where
    tLog t = apply0Tensor c_THDoubleTensor_log (tdDim tensor) t

td_lgamma :: TensorDouble -> TensorDouble
td_lgamma tensor = unsafePerformIO $ apply0_ tLgamma tensor
  where
    tLgamma t = apply0Tensor c_THDoubleTensor_lgamma (tdDim tensor) t

-- ----------------------------------------
-- c* cadd, cmul, cdiv, cpow, ...
-- ----------------------------------------

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

checkdim t src fun =
  unless ((tdDim t) == (tdDim src)) $ error ("Mismatched " ++ fun ++ " dimensions")

-- cadd = z <- y + scalar * x, z value discarded
-- allocate r_ for the user instead of taking it as an argument
td_cadd :: TensorDouble -> Double -> TensorDouble -> TensorDouble
td_cadd t scale src = unsafePerformIO $ do
  checkdim t src "cadd"
  apply2 ((swap1 c_THDoubleTensor_cadd) scaleC) t src
  where scaleC = realToFrac scale

td_csub :: TensorDouble -> Double -> TensorDouble -> TensorDouble
td_csub t scale src = unsafePerformIO $ do
  checkdim t src "csub"
  apply2 ((swap1 c_THDoubleTensor_csub) scaleC) t src
  where scaleC = realToFrac scale

td_cmul :: TensorDouble -> TensorDouble -> TensorDouble
td_cmul t src = unsafePerformIO $ do
  checkdim t src "cmul"
  apply2 c_THDoubleTensor_cmul t src

td_cpow :: TensorDouble -> TensorDouble -> TensorDouble
td_cpow t src = unsafePerformIO $ do
  checkdim t src "cpow"
  apply2 c_THDoubleTensor_cpow t src

td_cdiv :: TensorDouble -> TensorDouble -> TensorDouble
td_cdiv t src = unsafePerformIO $ do
  checkdim t src "cdiv"
  apply2 c_THDoubleTensor_cdiv t src

td_clshift :: TensorDouble -> TensorDouble -> TensorDouble
td_clshift t src = unsafePerformIO $ do
  checkdim t src "clshift"
  apply2 c_THDoubleTensor_clshift t src

td_crshift :: TensorDouble -> TensorDouble -> TensorDouble
td_crshift t src = unsafePerformIO $ do
  checkdim t src "crshift"
  apply2 c_THDoubleTensor_crshift t src

td_cfmod :: TensorDouble -> TensorDouble -> TensorDouble
td_cfmod t src = unsafePerformIO $ do
  checkdim t src "cfmod"
  apply2 c_THDoubleTensor_cfmod t src

td_cremainder :: TensorDouble -> TensorDouble -> TensorDouble
td_cremainder t src = unsafePerformIO $ do
  checkdim t src "cremainder"
  apply2 c_THDoubleTensor_cremainder t src

td_cbitand :: TensorDouble -> TensorDouble -> TensorDouble
td_cbitand  t src = unsafePerformIO $ do
  checkdim t src "cbitand"
  apply2 c_THDoubleTensor_cbitand t src

td_cbitor :: TensorDouble -> TensorDouble -> TensorDouble
td_cbitor  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitor t src

td_cbitxor :: TensorDouble -> TensorDouble -> TensorDouble
td_cbitxor t src = unsafePerformIO $ do
  checkdim t src "cbitxor"
  apply2 c_THDoubleTensor_cbitxor t src

td_addcmul :: TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addcmul t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcmul) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

td_addcdiv :: TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addcdiv t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcdiv) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

td_addmv :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addmv beta t alpha src1 src2 = unsafePerformIO $ do
  case (dim1, dim2, dimt) of
    (D2 _, D1 _, D1 _) -> pure ()
    _ -> error "Expected: 1D vector + 2D matrix x 1D vector"
  unless (d2 dim1 ^. _2 == d1 dim2) $ error "Matrix x vector dimension mismatch"
  unless (d2 dim1 ^. _1 == d1 dimt) $ error "Incorrect dimension for added vector"
  apply3 ((swap3 c_THDoubleTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)
    (dim1, dim2, dimt) = (tdDim src1, tdDim src2, tdDim t)

td_addmv_fast :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addmv_fast beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

td_mv :: TensorDouble -> TensorDouble -> TensorDouble
td_mv mat vec =
  td_addmv 0.0 zero 1.0 mat vec
  where
    zero = td_new $ (D1 ((^. _1) . d2 . tdDim $ mat))

td_mv_fast :: TensorDouble -> TensorDouble -> TensorDouble
td_mv_fast mat vec =
  td_addmv_fast 0.0 zero 1.0 mat vec
  where
    zero = td_new $ (D1 ((^. _1) . d2 . tdDim $ mat))

td_addmm :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addmm beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmm) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

td_addbmm :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

td_baddbmm :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_baddbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_baddbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

td_match :: TensorDouble -> TensorDouble -> Double -> TensorDouble
td_match m1 m2 gain = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_match) gainC) m1 m2
  where
    gainC = realToFrac gain
    swap fun gain b c d = fun b c d gain

td_numel :: TensorDouble -> Int
td_numel t = unsafePerformIO $ do
  result <- apply0_ c_THDoubleTensor_numel t
  pure $ fromIntegral result

-- TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_max :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_max t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_max t dimension keepdim

-- TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_min :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_min t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_min t dimension keepdim

-- TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
td_kthvalue :: TensorDouble -> Int -> Int -> Bool -> (TensorDouble, TensorLong)
td_kthvalue t k dimension keepdim = unsafePerformIO $
  ret2 ((swap c_THDoubleTensor_kthvalue) kC) t dimension keepdim
  where
    swap fun a b c d e f = fun b c d a e f -- curry k (4th argument)
    kC = fromIntegral k

-- TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_mode :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_mode t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_mode t dimension keepdim

-- TH_API void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_median :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_median t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_median t dimension keepdim

-- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
td_sum :: TensorDouble -> Int -> Bool -> TensorDouble
td_sum t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_sum) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
td_prod :: TensorDouble -> Int -> Bool -> TensorDouble
td_prod t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_prod) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
td_cumsum :: TensorDouble -> Int -> TensorDouble
td_cumsum t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumsum) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
td_cumprod :: TensorDouble -> Int -> TensorDouble
td_cumprod t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumprod) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);
td_sign :: TensorDouble -> TensorDouble
td_sign t = unsafePerformIO $ do
  apply1 c_THDoubleTensor_sign t

-- TH_API accreal THTensor_(trace)(THTensor *t);
td_trace :: TensorDouble -> Double
td_trace t = realToFrac $ unsafePerformIO $ do
  apply0_ c_THDoubleTensor_trace t

-- TH_API void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
td_cross :: TensorDouble -> TensorDouble -> Int -> TensorDouble
td_cross a b dimension = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_cross) dimensionC) a b
  where
    dimensionC = fromIntegral dimension
    swap fun a b c d = fun b c d a

-- TH_API void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
td_cmax :: TensorDouble -> TensorDouble -> TensorDouble
td_cmax t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmax t src

-- TH_API void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
td_cmin :: TensorDouble -> TensorDouble -> TensorDouble
td_cmin t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmin t src

-- -- TH_API void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
-- cmaxValue :: TensorDouble -> TensorDouble -> Double -> TensorDouble
-- cmaxValue t src value = unsafePerformIO $
--   apply2 c_THDoubleTensor_cmaxValue t src
--   where
--     swap 


-- TH_API void THTensor_(cminValue)(THTensor *r, THTensor *t, real value);

-- TH_API void THTensor_(zeros)(THTensor *r_, THLongStorage *size);
-- TH_API void THTensor_(zerosLike)(THTensor *r_, THTensor *input);
-- TH_API void THTensor_(ones)(THTensor *r_, THLongStorage *size);
-- TH_API void THTensor_(onesLike)(THTensor *r_, THTensor *input);
-- TH_API void THTensor_(diag)(THTensor *r_, THTensor *t, int k);

-- TH_API void THTensor_(eye)(THTensor *r_, long n, long m);
td_eye :: Word -> Word -> TensorDouble
td_eye d1 d2 = unsafePerformIO $ do
  withForeignPtr (tdTensor res)
    (\r_ -> do
        c_THDoubleTensor_eye r_ d1C d2C
        pure r_
    )
  pure res
  where
    res = td_new (D2 (d1, d2))
    d1C = fromIntegral d1
    d2C = fromIntegral d2

-- TH_API void THTensor_(arange)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
-- TH_API void THTensor_(range)(THTensor *r_, accreal xmin, accreal xmax, accreal step);
-- TH_API void THTensor_(randperm)(THTensor *r_, THGenerator *_generator, long n);

-- TH_API void THTensor_(reshape)(THTensor *r_, THTensor *t, THLongStorage *size);
-- TH_API void THTensor_(sort)(THTensor *rt_, THLongTensor *ri_, THTensor *t, int dimension, int descendingOrder);
-- TH_API void THTensor_(topk)(THTensor *rt_, THLongTensor *ri_, THTensor *t, long k, int dim, int dir, int sorted);
-- TH_API void THTensor_(tril)(THTensor *r_, THTensor *t, long k);
-- TH_API void THTensor_(triu)(THTensor *r_, THTensor *t, long k);
-- TH_API void THTensor_(cat)(THTensor *r_, THTensor *ta, THTensor *tb, int dimension);
-- TH_API void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension);

-- TH_API int THTensor_(equal)(THTensor *ta, THTensor *tb);
td_equal :: TensorDouble -> TensorDouble -> Bool
td_equal t1 t2 = unsafePerformIO $
  withForeignPtr (tdTensor t1)
    (\t1c ->
        withForeignPtr (tdTensor t2)
        (\t2c -> pure $ (c_THDoubleTensor_equal t1c t2c) == 1
        )
    )

-- TH_API void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);

-- TH_API void THTensor_(round)(THTensor *r_, THTensor *t);
td_round tensor = unsafePerformIO $ apply0_ tround tensor
  where
    tround t = apply0Tensor c_THDoubleTensor_round (tdDim tensor) t

-- -- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);

test = do
  -- check exception case
  let (m, v) = (td_init (D2 (3, 2)) 3.0 , td_init (D1 2) 2.0)
  disp m
  disp v
  disp $ td_mv m v

  let (m, v) = (td_init (D3 (1, 3, 2)) 3.0 , td_init (D1 2) 2.0)
  disp $ td_mv m v
  print "Done"
  pure () :: IO ()
