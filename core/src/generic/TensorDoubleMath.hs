module TensorDoubleMath (

  fillCopy_,
  fillMutate_,

  addConst,
  subConst,
  mulConst,
  divConst,

  (^+),
  (^-),
  (^*),
  (^/),

  dot,

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
  lgamma,

  cadd,
  csub,
  cmul,
  cpow,
  cdiv,
  clshift,
  crshift,
  cfmod,
  cremainder,
  cbitand,
  cbitor,
  cbitxor,
  addcmul,
  addcdiv,
  addmv,
  (!*),
  addmm,
  addbmm,
  baddbmm,
  match,
  numel,
  maxT,
  minT,
  kthvalue,
  mode,
  median,
  sumT,
  prod,
  cumsum,
  cumprod,
  sign,
  trace,
  cross
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import System.IO.Unsafe (unsafePerformIO)

import TensorDouble
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

-- |Generalize non-mutating collapse of a tensor to a constant or another tensor
apply0_ :: (Ptr CTHDoubleTensor -> a) -> TensorDouble_ -> IO a
apply0_ operation tensor = do
  withForeignPtr (tdTensor tensor) (\t -> pure $ operation t)

-- |Wrapper to apply tensor -> tensor non-mutating operation
apply0Tensor op resDim t = unsafePerformIO $ do
  let res = tensorNew_ resDim
  withForeignPtr (tdTensor res) (\r_ -> op r_ t)
  pure res

-- usually this is 1 mutation arg + 2 parameter args
type Raw3Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- usually this is 1 mutation arg + 3 parameter args
type Raw4Arg = Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor ->Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

apply2 :: Raw3Arg -> TensorDouble_ -> TensorDouble_ -> IO TensorDouble_
apply2 fun t src = do
  let r_ = tensorNew_ (tdDim t)
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

apply3 :: Raw4Arg -> TensorDouble_ -> TensorDouble_ -> TensorDouble_ -> IO TensorDouble_
apply3 fun t src1 src2 = do
  let r_ = tensorNew_ (tdDim t)
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

ret2 :: Ret2Fun -> TensorDouble_ -> Int -> Bool -> IO (TensorDouble_, TensorLong)
ret2 fun t dimension keepdim = do
  let values_ = tensorNew_ (tdDim t)
  let indices_ = tensorNewLong (tdDim t)
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
  let r_ = tensorNew_ (tdDim t)
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

(^+) = addConst
(^-) = subConst
(^*) = mulConst
(^/) = divConst

-- ----------------------------------------
-- Linear algebra
-- ----------------------------------------

dot :: TensorDouble_ -> TensorDouble_ -> Double
dot t src = realToFrac $ unsafePerformIO $ do
  withForeignPtr (tdTensor t)
    (\tPtr -> withForeignPtr (tdTensor src)
      (\srcPtr ->
          pure $ c_THDoubleTensor_dot tPtr srcPtr
      )
    )

-- ----------------------------------------
-- Collapse to constant operations
-- ----------------------------------------

minAll :: TensorDouble_ -> Double
minAll tensor = unsafePerformIO $ apply0_ tMinAll tensor
  where
    tMinAll t = realToFrac $ c_THDoubleTensor_minall t

maxAll :: TensorDouble_ -> Double
maxAll tensor = unsafePerformIO $ apply0_ tMaxAll tensor
  where
    tMaxAll t = realToFrac $ c_THDoubleTensor_maxall t

medianAll :: TensorDouble_ -> Double
medianAll tensor = unsafePerformIO $ apply0_ tMedianAll tensor
  where
    tMedianAll t = realToFrac $ c_THDoubleTensor_medianall t

sumAll :: TensorDouble_ -> Double
sumAll tensor = unsafePerformIO $ apply0_ tSumAll tensor
  where
    tSumAll t = realToFrac $ c_THDoubleTensor_sumall t

prodAll :: TensorDouble_ -> Double
prodAll tensor = unsafePerformIO $ apply0_ tProdAll tensor
  where
    tProdAll t = realToFrac $ c_THDoubleTensor_prodall t

meanAll :: TensorDouble_ -> Double
meanAll tensor = unsafePerformIO $ apply0_ tMeanAll tensor
  where
    tMeanAll t = realToFrac $ c_THDoubleTensor_meanall t

-- ----------------------------------------
-- Tensor to Tensor transformation
-- ----------------------------------------

neg :: TensorDouble_ -> TensorDouble_
neg tensor = unsafePerformIO $ apply0_ tNeg tensor
  where
    tNeg t = apply0Tensor c_THDoubleTensor_neg (tdDim tensor) t

absT :: TensorDouble_ -> TensorDouble_
absT tensor = unsafePerformIO $ apply0_ tAbs tensor
  where
    tAbs t = apply0Tensor c_THDoubleTensor_abs (tdDim tensor) t

sigmoid :: TensorDouble_ -> TensorDouble_
sigmoid tensor = unsafePerformIO $ apply0_ tSigmoid tensor
  where
    tSigmoid t = apply0Tensor c_THDoubleTensor_sigmoid (tdDim tensor) t

logT :: TensorDouble_ -> TensorDouble_
logT tensor = unsafePerformIO $ apply0_ tLog tensor
  where
    tLog t = apply0Tensor c_THDoubleTensor_log (tdDim tensor) t

lgamma :: TensorDouble_ -> TensorDouble_
lgamma tensor = unsafePerformIO $ apply0_ tLgamma tensor
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

-- cadd = z <- y + scalar * x, z value discarded
-- allocate r_ for the user instead of taking it as an argument
cadd :: TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_
cadd t scale src = unsafePerformIO $
  apply2 ((swap1 c_THDoubleTensor_cadd) scaleC) t src
  where scaleC = realToFrac scale

csub :: TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_
csub t scale src = unsafePerformIO $ do
  apply2 ((swap1 c_THDoubleTensor_csub) scaleC) t src
  where scaleC = realToFrac scale

cmul :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cmul t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cmul t src

cpow :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cpow t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cpow t src

cdiv :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cdiv t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cdiv t src

clshift :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
clshift t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_clshift t src

crshift :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
crshift t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_crshift t src

cfmod :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cfmod t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cfmod t src

cremainder :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cremainder t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cremainder t src

cbitand :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cbitand  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitand t src

cbitor :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cbitor  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitor t src

cbitxor :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cbitxor t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitxor t src

addcmul :: TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_ -> TensorDouble_
addcmul t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcmul) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

addcdiv :: TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_ -> TensorDouble_
addcdiv t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcdiv) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble

addmv :: Double -> TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_ -> TensorDouble_
addmv beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

(!*) :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
mat !* vec =
  addmv 1.0 zero 1.0 mat vec
  where
    zero = tensorNew_ $ tdDim vec -- TODO - more efficient version w/o allocaiton?

addmm :: Double -> TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_ -> TensorDouble_
addmm beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmm) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

addbmm :: Double -> TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_ -> TensorDouble_
addbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

baddbmm :: Double -> TensorDouble_ -> Double -> TensorDouble_ -> TensorDouble_ -> TensorDouble_
baddbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_baddbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)

match :: TensorDouble_ -> TensorDouble_ -> Double -> TensorDouble_
match m1 m2 gain = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_match) gainC) m1 m2
  where
    gainC = realToFrac gain
    swap fun gain b c d = fun b c d gain

numel :: TensorDouble_ -> Int
numel t = unsafePerformIO $ do
  result <- apply0_ c_THDoubleTensor_numel t
  pure $ fromIntegral result

-- TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
maxT :: TensorDouble_ -> Int -> Bool -> (TensorDouble_, TensorLong)
maxT t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_max t dimension keepdim

-- TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
minT :: TensorDouble_ -> Int -> Bool -> (TensorDouble_, TensorLong)
minT t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_min t dimension keepdim

-- TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
kthvalue :: TensorDouble_ -> Int -> Int -> Bool -> (TensorDouble_, TensorLong)
kthvalue t k dimension keepdim = unsafePerformIO $
  ret2 ((swap c_THDoubleTensor_kthvalue) kC) t dimension keepdim
  where
    swap fun a b c d e f = fun b c d a e f -- curry k (4th argument)
    kC = fromIntegral k

-- TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
mode :: TensorDouble_ -> Int -> Bool -> (TensorDouble_, TensorLong)
mode t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_mode t dimension keepdim

-- TH_API void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
median :: TensorDouble_ -> Int -> Bool -> (TensorDouble_, TensorLong)
median t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_median t dimension keepdim

-- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
sumT :: TensorDouble_ -> Int -> Bool -> TensorDouble_
sumT t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_sum) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
prod :: TensorDouble_ -> Int -> Bool -> TensorDouble_
prod t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_prod) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0

-- TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
cumsum :: TensorDouble_ -> Int -> TensorDouble_
cumsum t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumsum) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
cumprod :: TensorDouble_ -> Int -> TensorDouble_
cumprod t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumprod) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension

-- TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);
sign :: TensorDouble_ -> TensorDouble_
sign t = unsafePerformIO $ do
  apply1 c_THDoubleTensor_sign t

-- TH_API accreal THTensor_(trace)(THTensor *t);
trace :: TensorDouble_ -> Double
trace t = realToFrac $ unsafePerformIO $ do
  apply0_ c_THDoubleTensor_trace t

-- TH_API void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
cross :: TensorDouble_ -> TensorDouble_ -> Int -> TensorDouble_
cross a b dimension = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_cross) dimensionC) a b
  where
    dimensionC = fromIntegral dimension
    swap fun a b c d = fun b c d a

-- TH_API void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
cmax :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cmax t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmax t src

-- TH_API void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
cmin :: TensorDouble_ -> TensorDouble_ -> TensorDouble_
cmin t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmin t src

-- -- TH_API void THTensor_(cmaxValue)(THTensor *r, THTensor *t, real value);
-- cmaxValue :: TensorDouble_ -> TensorDouble_ -> Double -> TensorDouble_
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

-- TH_API void THTensor_(ltValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(leValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(gtValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(geValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(neValue)(THByteTensor *r_, THTensor* t, real value);
-- TH_API void THTensor_(eqValue)(THByteTensor *r_, THTensor* t, real value);

