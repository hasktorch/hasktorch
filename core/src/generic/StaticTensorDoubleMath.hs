module StaticTensorDoubleMath (

  tds_fill,
  tds_fill_,

  tds_addConst,
  tds_subConst,
  tds_mulConst,
  tds_divConst,

  (^+),
  (^-),
  (^*),
  (^/),

  tds_dot,

  tds_minAll,
  tds_maxAll,
  tds_medianAll,
  tds_sumAll,
  tds_prodAll,
  tds_meanAll,

  tds_neg,
  tds_absT,
  tds_sigmoid,
  tds_logT,
  tds_lgamma,

  tds_cadd,
  tds_csub,
  tds_cmul,
  tds_cpow,
  tds_cdiv,
  tds_clshift,
  tds_crshift,
  tds_cfmod,
  tds_cremainder,
  tds_cbitand,
  tds_cbitor,
  tds_cbitxor,
  tds_addcmul,
  tds_addcdiv,
  tds_addmv,
--  (!*),
  tds_addmm,
  tds_addbmm,
  tds_baddbmm,
  tds_match,
  tds_numel,
  tds_maxT,
  tds_minT,
  tds_kthvalue,
  tds_mode,
  tds_median,
  tds_sumT,
  tds_prod,
  tds_cumsum,
  tds_cumprod,
  tds_sign,
  tds_trace,
  tds_cross
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import System.IO.Unsafe (unsafePerformIO)

import StaticTensorDouble
import TensorLong
import TensorRaw
import TensorTypes

import THDoubleTensor
import THDoubleTensor
import THDoubleTensorMath
import THTypes

-- ----------------------------------------
-- Foreign pointer application helper functions
-- ----------------------------------------

apply1_ :: (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO a)
     -> (TDS n d) -> p -> (TDS n d)
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

-- |Generalize non-mutating collapse of a tensor to a constant or another tensor
apply0_ :: (Ptr CTHDoubleTensor -> a) -> (TDS n d) -> IO a
apply0_ operation tensor = do
  withForeignPtr (tdsTensor tensor) (\t -> pure $ operation t)

-- |Wrapper to apply tensor -> tensor non-mutating operation
apply0Tensor :: (Ptr CTHDoubleTensor -> t -> IO a) -> TensorDim Word -> t
  -> (TDS n d)
apply0Tensor op resDim t = unsafePerformIO $ do
  let res = tds_new
  withForeignPtr (tdsTensor res) (\r_ -> op r_ t)
  pure res

-- usually this is 1 mutation arg + 2 parameter args
type Raw3Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- usually this is 1 mutation arg + 3 parameter args
type Raw4Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor ->Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

apply2 :: Raw3Arg -> (TDS n d) -> (TDS n d) -> IO (TDS n d)
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

apply3 :: Raw4Arg -> (TDS n d) -> (TDS n d) -> (TDS n d) -> IO (TDS n d)
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

type Ret2Fun =
  Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

ret2 :: Ret2Fun -> (TDS n d) -> Int -> Bool -> IO ((TDS n d), TensorLong)
ret2 fun t dimension keepdim = do
  let values_ = tds_new
  let indices_ = tl_new (tdsDim t)
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

-- ----------------------------------------
-- Tensor fill operations
-- ----------------------------------------

tds_fill :: Real a => a -> (TDS n d) -> (TDS n d)
tds_fill value tensor = unsafePerformIO $
  withForeignPtr(tdsTensor nt) (\t -> do
                                  fillRaw value t
                                  pure nt
                              )
  where nt = tds_new

tds_fill_ :: Real a => a -> (TDS n d) -> IO ()
tds_fill_ value tensor =
  withForeignPtr(tdsTensor tensor) (\t -> fillRaw value t)

-- ----------------------------------------
-- Tensor-constant operations to constant operations
-- ----------------------------------------

tds_addConst :: (TDS n d) -> Double -> (TDS n d)
tds_addConst mtx val = apply1_ tAdd mtx val
  where
    tAdd r_ t = c_THDoubleTensor_add r_ t (realToFrac val)

tds_subConst :: (TDS n d) -> Double -> (TDS n d)
tds_subConst mtx val = apply1_ tSub mtx val
  where
    tSub r_ t = c_THDoubleTensor_sub r_ t (realToFrac val)

tds_mulConst :: (TDS n d) -> Double -> (TDS n d)
tds_mulConst mtx val = apply1_ tMul mtx val
  where
    tMul r_ t = c_THDoubleTensor_mul r_ t (realToFrac val)

tds_divConst :: (TDS n d) -> Double -> (TDS n d)
tds_divConst mtx val = apply1_ tDiv mtx val
  where
    tDiv r_ t = c_THDoubleTensor_div r_ t (realToFrac val)

(^+) = tds_addConst
(^-) = tds_subConst
(^*) = tds_mulConst
(^/) = tds_divConst

-- ----------------------------------------
-- Linear algebra
-- ----------------------------------------

tds_dot :: (TDS n d) -> (TDS n d) -> Double
tds_dot t src = realToFrac $ unsafePerformIO $ do
  withForeignPtr (tdsTensor t)
    (\tPtr -> withForeignPtr (tdsTensor src)
      (\srcPtr ->
          pure $ c_THDoubleTensor_dot tPtr srcPtr
      )
    )

-- ----------------------------------------
-- Collapse to constant operations
-- ----------------------------------------

tds_minAll :: (TDS n d) -> Double
tds_minAll tensor = unsafePerformIO $ apply0_ tMinAll tensor
  where
    tMinAll t = realToFrac $ c_THDoubleTensor_minall t

tds_maxAll :: (TDS n d) -> Double
tds_maxAll tensor = unsafePerformIO $ apply0_ tMaxAll tensor
  where
    tMaxAll t = realToFrac $ c_THDoubleTensor_maxall t

tds_medianAll :: (TDS n d) -> Double
tds_medianAll tensor = unsafePerformIO $ apply0_ tMedianAll tensor
  where
    tMedianAll t = realToFrac $ c_THDoubleTensor_medianall t

tds_sumAll :: (TDS n d) -> Double
tds_sumAll tensor = unsafePerformIO $ apply0_ tSumAll tensor
  where
    tSumAll t = realToFrac $ c_THDoubleTensor_sumall t

tds_prodAll :: (TDS n d) -> Double
tds_prodAll tensor = unsafePerformIO $ apply0_ tProdAll tensor
  where
    tProdAll t = realToFrac $ c_THDoubleTensor_prodall t

tds_meanAll :: (TDS n d) -> Double
tds_meanAll tensor = unsafePerformIO $ apply0_ tMeanAll tensor
  where
    tMeanAll t = realToFrac $ c_THDoubleTensor_meanall t

-- ----------------------------------------
-- Tensor to Tensor transformation
-- ----------------------------------------

tds_neg :: (TDS n d) -> (TDS n d)
tds_neg tensor = unsafePerformIO $ apply0_ tNeg tensor
  where
    tNeg t = apply0Tensor c_THDoubleTensor_neg (tdsDim tensor) t

tds_absT :: (TDS n d) -> (TDS n d)
tds_absT tensor = unsafePerformIO $ apply0_ tAbs tensor
  where
    tAbs t = apply0Tensor c_THDoubleTensor_abs (tdsDim tensor) t

tds_sigmoid :: (TDS n d) -> (TDS n d)
tds_sigmoid tensor = unsafePerformIO $ apply0_ tSigmoid tensor
  where
    tSigmoid t = apply0Tensor c_THDoubleTensor_sigmoid (tdsDim tensor) t

tds_logT :: (TDS n d) -> (TDS n d)
tds_logT tensor = unsafePerformIO $ apply0_ tLog tensor
  where
    tLog t = apply0Tensor c_THDoubleTensor_log (tdsDim tensor) t

tds_lgamma :: (TDS n d) -> (TDS n d)
tds_lgamma tensor = unsafePerformIO $ apply0_ tLgamma tensor
  where
    tLgamma t = apply0Tensor c_THDoubleTensor_lgamma (tdsDim tensor) t

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

-- cadd = z <- y + scalar * x, z value discarded
-- allocate r_ for the user instead of taking it as an argument
tds_cadd :: (TDS n d) -> Double -> (TDS n d) -> (TDS n d)
tds_cadd t scale src = unsafePerformIO $
  apply2 ((swap1 c_THDoubleTensor_cadd) scaleC) t src
  where scaleC = realToFrac scale

tds_csub :: (TDS n d) -> Double -> (TDS n d) -> (TDS n d)
tds_csub t scale src = unsafePerformIO $ do
  apply2 ((swap1 c_THDoubleTensor_csub) scaleC) t src
  where scaleC = realToFrac scale

tds_cmul :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cmul t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cmul t src

tds_cpow :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cpow t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cpow t src

tds_cdiv :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cdiv t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cdiv t src

tds_clshift :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_clshift t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_clshift t src

tds_crshift :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_crshift t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_crshift t src

tds_cfmod :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cfmod t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cfmod t src

tds_cremainder :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cremainder t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cremainder t src

tds_cbitand :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cbitand  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitand t src

tds_cbitor :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cbitor  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitor t src

tds_cbitxor :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cbitxor t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitxor t src

tds_addcmul :: (TDS n d) -> Double -> (TDS n d) -> (TDS n d) -> (TDS n d)
tds_addcmul t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcmul) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

tds_addcdiv :: (TDS n d) -> Double -> (TDS n d) -> (TDS n d) -> (TDS n d)
tds_addcdiv t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcdiv) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

tds_addmv :: Double -> (TDS n d) -> Double -> (TDS n d) -> (TDS n d) -> (TDS n d)
tds_addmv beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

-- (!*) :: (TDS n d) -> (TDS n d) -> (TDS n d)
-- mat !* vec =
--   tds_addmv 1.0 zero 1.0 mat vec
--   where
--     zero = tds_new

tds_addmm :: Double -> (TDS n d) -> Double -> (TDS n d) -> (TDS n d) -> (TDS n d)
tds_addmm beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmm) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

tds_addbmm :: Double -> (TDS n d) -> Double -> (TDS n d) -> (TDS n d) -> (TDS n d)
tds_addbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

tds_baddbmm :: Double -> (TDS n d) -> Double -> (TDS n d) -> (TDS n d) -> (TDS n d)
tds_baddbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_baddbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

tds_match :: (TDS n d) -> (TDS n d) -> Double -> (TDS n d)
tds_match m1 m2 gain = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_match) gainC) m1 m2
  where
    gainC = realToFrac gain
    swap fun gain b c d = fun b c d gain

tds_numel :: (TDS n d) -> Int
tds_numel t = unsafePerformIO $ do
  result <- apply0_ c_THDoubleTensor_numel t
  pure $ fromIntegral result

-- TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tds_maxT :: (TDS n d) -> Int -> Bool -> ((TDS n d), TensorLong)
tds_maxT t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_max t dimension keepdim

-- TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tds_minT :: (TDS n d) -> Int -> Bool -> ((TDS n d), TensorLong)
tds_minT t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_min t dimension keepdim

-- TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
tds_kthvalue :: (TDS n d) -> Int -> Int -> Bool -> ((TDS n d), TensorLong)
tds_kthvalue t k dimension keepdim = unsafePerformIO $
  ret2 ((swap c_THDoubleTensor_kthvalue) kC) t dimension keepdim
  where
    swap fun a b c d e f = fun b c d a e f -- curry k (4th argument)
    kC = fromIntegral k

-- TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tds_mode :: (TDS n d) -> Int -> Bool -> ((TDS n d), TensorLong)
tds_mode t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_mode t dimension keepdim

-- TH_API void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tds_median :: (TDS n d) -> Int -> Bool -> ((TDS n d), TensorLong)
tds_median t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_median t dimension keepdim

-- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
tds_sumT :: (TDS n d) -> Int -> Bool -> (TDS n d)
tds_sumT t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_sum) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
tds_prod :: (TDS n d) -> Int -> Bool -> (TDS n d)
tds_prod t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_prod) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
tds_cumsum :: (TDS n d) -> Int -> (TDS n d)
tds_cumsum t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumsum) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
tds_cumprod :: (TDS n d) -> Int -> (TDS n d)
tds_cumprod t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumprod) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);
tds_sign :: (TDS n d) -> (TDS n d)
tds_sign t = unsafePerformIO $ do
  apply1 c_THDoubleTensor_sign t

-- TH_API accreal THTensor_(trace)(THTensor *t);
tds_trace :: (TDS n d) -> Double
tds_trace t = realToFrac $ unsafePerformIO $ do
  apply0_ c_THDoubleTensor_trace t

-- TH_API void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
tds_cross :: (TDS n d) -> (TDS n d) -> Int -> (TDS n d)
tds_cross a b dimension = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_cross) dimensionC) a b
  where
    dimensionC = fromIntegral dimension
    swap fun a b c d = fun b c d a

-- TH_API void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
tds_cmax :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cmax t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmax t src

-- TH_API void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
tds_cmin :: (TDS n d) -> (TDS n d) -> (TDS n d)
tds_cmin t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmin t src


