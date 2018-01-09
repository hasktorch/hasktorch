{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Core.Tensor.Static.FloatMath
  () where
{-
  ( (^+^)
  , (^-^)
  , (^*^)
  , (^/^)
  , (!*)
  , (!*!)
  , (^+)
  , (^-)
  , (+^)
  , (-^)
  , (^*)
  , (^/)
  , (*^)
  , (/^)
  , (<.>)

  , tfs_fill
  , tfs_fill_

  , tfs_addConst
  , tfs_subConst
  , tfs_mulConst
  , tfs_divConst

  , tfs_dot

  , tfs_minAll
  , tfs_maxAll
  , tfs_medianAll
  , tfs_sumAll
  , tfs_prodAll
  , tfs_meanAll

  , tfs_neg
  , tfs_cinv
  , tfs_ltTensorT
  , tfs_leTensorT
  , tfs_gtTensorT
  , tfs_geTensorT
  , tfs_neTensorT
  , tfs_eqTensorT
  , tfs_abs

  , tfs_sigmoid
  , tfs_log
  , tfs_lgamma
  , tfs_log1p
  , tfs_exp
  , tfs_acos
  , tfs_cosh
  , tfs_sin
  , tfs_asin
  , tfs_sinh
  , tfs_tan
  , tfs_atan
  -- , tfs_atan2
  , tfs_tanh
  , tfs_pow
  , tfs_tpow
  , tfs_sqrt
  , tfs_rsqrt
  , tfs_ceil
  , tfs_floor
  , tfs_round

  , tfs_cadd
  , tfs_csub
  , tfs_cmul
  , tfs_square
  , tfs_cpow
  , tfs_cdiv
  , tfs_clshift
  , tfs_crshift
  , tfs_cfmod
  , tfs_cremainder
  , tfs_cbitand
  , tfs_cbitor
  , tfs_cbitxor
  -- , tfs_addcmul
  -- , tfs_addcdiv
  , tfs_addmv
  , tfs_mv

  , tfs_addmm
  , tfs_addr
  , tfs_outer
  , tfs_addbmm
  , tfs_baddbmm
  , tfs_match
  , tfs_numel
  , tfs_max
  , tfs_min
  , tfs_kthvalue
  , tfs_mode
  , tfs_median
  , tfs_sum
  , tfs_colsum
  , tfs_rowsum
  , tfs_prod
  , tfs_cumsum
  , tfs_cumprod
  , tfs_sign
  , tfs_trace
  , tfs_cross

  , tfs_equal

  , tfs_cat
  , tfs_diag

  ) where

import Control.Monad.Managed
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Foreign (Ptr)
import Foreign.C.Types (CLong, CFloat, CInt)
import Foreign.ForeignPtr ( ForeignPtr, withForeignPtr, newForeignPtr )
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import Torch.Core.Tensor.Dynamic.Long
import THTypes
import THFloatTensor
import THFloatTensorMath
import Torch.Core.Tensor.Static.Float

{- Operators -}

-- |Experimental num instance for static tensors
instance SingI d => Num (TensorFloatStatic d) where
  (+) t1 t2 = tfs_cadd t1 1.0 t2
  (-) t1 t2 = tfs_csub t1 1.0  t2
  (*) t1 t2 = tfs_cmul t1 t2
  abs t = tfs_abs t
  signum t = error "signum not defined for tensors"
  fromInteger t = error "signum not defined for tensors"

(!*) :: (KnownNat c, KnownNat r) => (TFS '[r, c]) -> (TFS '[c]) -> (TFS '[r])
(!*) m v = tfs_mv m v

(!*!) :: (KnownNat a, KnownNat b, KnownNat c) =>
  TFS '[a, b] -> TFS '[b, c] -> TFS '[a,c]
(!*!) m1 m2 = tfs_addmm 1.0 tfs_new 1.0 m1 m2

(^+^) t1 t2 = tfs_cadd t1 1.0 t2
(^-^) t1 t2 = tfs_csub t1 1.0 t2
(^*^) t1 t2 = tfs_cmul t1 t2
(^/^) t1 t2 = tfs_cdiv t1 t2

(^+) :: (Real p, SingI d) => TFS d -> p -> TFS d
(^+) = tfs_addConst

(^-) :: (Real p, SingI d) => TFS d -> p -> TFS d
(^-) = tfs_subConst

(^*) :: (SingI d, Real p) => TFS d -> p -> TFS d
(^*) = tfs_mulConst

(^/) :: (SingI d, Real p) => TFS d -> p -> TFS d
(^/) = tfs_divConst

(+^) :: (Real p, SingI d) => p -> TFS d -> TFS d
(+^) = flip tfs_addConst

(-^) :: (Real p, SingI d) => p -> TFS d -> TFS d
(-^) val t = tfs_addConst (tfs_neg t) val

(*^) :: (SingI d, Real p) => p -> TFS d -> TFS d
(*^) = flip tfs_mulConst

(/^) :: (SingI d, Real p) => p -> TFS d -> TFS d
(/^) val t = tfs_mulConst (tfs_cinv t) val

(<.>) t1 t2 = tfs_dot t1 t2

{- Helper functions -}

-- |Generalize non-mutating collapse of a tensor to a constant or another tensor
apply0_ :: (Ptr CTHFloatTensor -> a) -> (TFS d) -> IO a
apply0_ operation tensor = do
  withForeignPtr (tdsTensor tensor) (\t -> pure $ operation t)

apply1_ :: SingI d => (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO a)
     -> (TFS d) -> p -> (TFS d)
apply1_ transformation mtx val = unsafePerformIO $ do
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor res)
    tPtr <- managed $ withForeignPtr (tdsTensor mtx)
    liftIO $ transformation rPtr tPtr
    liftIO (pure ())
  pure res
  where
    res = tfs_cloneDim mtx
{-# NOINLINE apply1_ #-}

tfs_fill :: (Real a, SingI d) => a -> p -> TensorFloatStatic d
tfs_fill value tensor = unsafePerformIO $
  withForeignPtr (tdsTensor nt) (\t -> do
                                  fillFloatRaw value t
                                  pure nt
                              )
  where nt = tfs_new
{-# NOINLINE tfs_fill #-}

tfs_fill_ :: Real a => a -> (TFS d) -> IO ()
tfs_fill_ value tensor =
  withForeignPtr(tdsTensor tensor) (\t -> fillFloatRaw value t)

tfs_addConst :: (SingI d, Real p) => TFS d -> p -> TFS d
tfs_addConst mtx val = apply1_ tAdd mtx val
  where
    tAdd r_ t = c_THFloatTensor_add r_ t (realToFrac val)

tfs_subConst :: (SingI d, Real p) => TFS d -> p -> TFS d
tfs_subConst mtx val = apply1_ tSub mtx val
  where
    tSub r_ t = c_THFloatTensor_sub r_ t (realToFrac val)

tfs_mulConst :: (SingI d, Real p) => TFS d -> p -> TFS d
tfs_mulConst mtx val = apply1_ tMul mtx val
  where
    tMul r_ t = c_THFloatTensor_mul r_ t (realToFrac val)

tfs_divConst :: (SingI d, Real p) => TFS d -> p -> TFS d
tfs_divConst mtx val = apply1_ tDiv mtx val
  where
    tDiv r_ t = c_THFloatTensor_div r_ t (realToFrac val)

tfs_dot :: (TFS d) -> (TFS d) -> Float
tfs_dot t src = realToFrac $ unsafePerformIO $ do
  with (do
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    srcPtr <- managed $ withForeignPtr (tdsTensor src)
    pure (tPtr, srcPtr))
    (\(tPtr, srcPtr) -> pure $ c_THFloatTensor_dot tPtr srcPtr)
{-# NOINLINE tfs_dot #-}

tfs_minAll :: (TFS d) -> Float
tfs_minAll tensor = unsafePerformIO $ apply0_ tMinAll tensor
  where
    tMinAll t = realToFrac $ c_THFloatTensor_minall t
{-# NOINLINE tfs_minAll #-}

tfs_maxAll :: (TFS d) -> Float
tfs_maxAll tensor = unsafePerformIO $ apply0_ tMaxAll tensor
  where
    tMaxAll t = realToFrac $ c_THFloatTensor_maxall t
{-# NOINLINE tfs_maxAll #-}

tfs_medianAll :: (TFS d) -> Float
tfs_medianAll tensor = unsafePerformIO $ apply0_ tMedianAll tensor
  where
    tMedianAll t = realToFrac $ c_THFloatTensor_medianall t
{-# NOINLINE tfs_medianAll #-}

tfs_sumAll :: (TFS d) -> Float
tfs_sumAll tensor = unsafePerformIO $ apply0_ tSumAll tensor
  where
    tSumAll t = realToFrac $ c_THFloatTensor_sumall t
{-# NOINLINE tfs_sumAll #-}

tfs_prodAll :: (TFS d) -> Float
tfs_prodAll tensor = unsafePerformIO $ apply0_ tProdAll tensor
  where
    tProdAll t = realToFrac $ c_THFloatTensor_prodall t
{-# NOINLINE tfs_prodAll #-}

tfs_meanAll :: (TFS d) -> Float
tfs_meanAll tensor = unsafePerformIO $ apply0_ tMeanAll tensor
  where
    tMeanAll t = realToFrac $ c_THFloatTensor_meanall t
{-# NOINLINE tfs_meanAll #-}

-- ----------------------------------------
-- Tensor to Tensor transformation
-- ----------------------------------------

-- |Wrapper to apply tensor -> tensor non-mutating operation
apply0Tensor
  :: SingI d => (Ptr CTHFloatTensor -> t -> IO a) -> t -> (TFS d)
apply0Tensor op t = unsafePerformIO $ do
  let res = tfs_new
  withForeignPtr (tdsTensor res) (\r_ -> op r_ t)
  pure res
{-# NOINLINE apply0Tensor #-}

-- |Returns a tensor with values negated
tfs_neg :: SingI d => TFS d -> TFS d
tfs_neg tensor = unsafePerformIO $ apply0_ tNeg tensor
  where
    tNeg t = apply0Tensor c_THFloatTensor_neg t
{-# NOINLINE tfs_neg #-}

tfs_cinv :: SingI d => TFS d -> TFS d
tfs_cinv tensor = unsafePerformIO $ apply0_ tInv tensor
  where
    tInv t = apply0Tensor c_THFloatTensor_cinv t
{-# NOINLINE tfs_cinv #-}

-- ----------------------------------------
-- Tensor vs. tensor comparison, retaining double type
-- ----------------------------------------

-- |Returns a tensor of 0.0 and 1.0 by comparing whether ta < tb for each value
tfs_ltTensorT :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_ltTensorT ta tb = unsafePerformIO $
  apply2 c_THFloatTensor_ltTensorT ta tb
{-# NOINLINE tfs_ltTensorT #-}

-- |Returns a tensor of 0.0 and 1.0 by comparing whether ta <= tb for each value
tfs_leTensorT :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_leTensorT ta tb = unsafePerformIO $
  apply2 c_THFloatTensor_leTensorT ta tb
{-# NOINLINE tfs_leTensorT #-}

-- |Returns a tensor of 0.0 and 1.0 by comparing whether ta > tb for each value
tfs_gtTensorT :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_gtTensorT ta tb = unsafePerformIO $
  apply2 c_THFloatTensor_gtTensorT ta tb
{-# NOINLINE tfs_gtTensorT #-}

-- |Returns a tensor of 0.0 and 1.0 by comparing whether ta >= tb for each value
tfs_geTensorT :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_geTensorT ta tb = unsafePerformIO $
  apply2 c_THFloatTensor_geTensorT ta tb
{-# NOINLINE tfs_geTensorT #-}

-- |Returns a tensor of 0.0 and 1.0 by comparing whether ta /= tb for each value
tfs_neTensorT :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_neTensorT ta tb = unsafePerformIO $
  apply2 c_THFloatTensor_neTensorT ta tb
{-# NOINLINE tfs_neTensorT #-}

-- |Returns a tensor of 0.0 and 1.0 by comparing whether ta == tb for each value
tfs_eqTensorT :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_eqTensorT ta tb = unsafePerformIO $
  apply2 c_THFloatTensor_eqTensorT ta tb
{-# NOINLINE tfs_eqTensorT #-}

-- ----------------------------------------
-- Additional transformations
-- ----------------------------------------

-- |Returns a tensor where each value of the input tensor is transformed as its
-- absolute value
tfs_abs :: SingI d => TFS d -> TFS d
tfs_abs t = unsafePerformIO $ apply0_ tAbs t
  where
    tAbs t' = apply0Tensor c_THFloatTensor_abs t'
{-# NOINLINE tfs_abs #-}

-- |Returns a t where each value of the input tensor is transformed as the
-- sigmoid of the value
tfs_sigmoid :: SingI d => TFS d -> TFS d
tfs_sigmoid t = unsafePerformIO $ apply0_ tSigmoid t
  where
    tSigmoid t' = apply0Tensor c_THFloatTensor_sigmoid t'
{-# NOINLINE tfs_sigmoid #-}

-- |Returns a tensor where each value of the input tensor is transformed as the
-- log of the value
tfs_log :: SingI d => TFS d -> TFS d
tfs_log t = unsafePerformIO $ apply0_ tLog t
  where
    tLog t' = apply0Tensor c_THFloatTensor_log t'
{-# NOINLINE tfs_log #-}

-- |Returns a tensor where each value of the input tensor is transformed as the
-- log gamma of the value
tfs_lgamma :: SingI d => TFS d -> TFS d
tfs_lgamma t = unsafePerformIO $ apply0_ tLgamma t
  where
    tLgamma t' = apply0Tensor c_THFloatTensor_lgamma t'
{-# NOINLINE tfs_lgamma #-}

-- |Returns a new tensor with the natural log of 1 + the elements
tfs_log1p :: SingI d => TFS d -> TFS d
tfs_log1p t = unsafePerformIO $ apply0_ tLog1p t
  where
    tLog1p t' = apply0Tensor c_THFloatTensor_log1p t'
{-# NOINLINE tfs_log1p #-}

-- |Returns a tensor where each value of the input tensor is transformed as the
-- exp of the value
tfs_exp :: SingI d => TFS d -> TFS d
tfs_exp t = unsafePerformIO $ apply0_ tExp t
  where
    tExp t' = apply0Tensor c_THFloatTensor_exp t'
{-# NOINLINE tfs_exp #-}

tfs_cos :: SingI d => TFS d -> TFS d
tfs_cos t = unsafePerformIO $ apply0_ tCos t
  where
    tCos t' = apply0Tensor c_THFloatTensor_cos t'
{-# NOINLINE tfs_cos #-}

tfs_acos :: SingI d => TFS d -> TFS d
tfs_acos t = unsafePerformIO $ apply0_ tAcos t
  where
    tAcos t' = apply0Tensor c_THFloatTensor_acos t'
{-# NOINLINE tfs_acos #-}

tfs_cosh :: SingI d => TFS d -> TFS d
tfs_cosh t = unsafePerformIO $ apply0_ tCosh t
  where
    tCosh t' = apply0Tensor c_THFloatTensor_cosh t'
{-# NOINLINE tfs_cosh #-}

tfs_sin :: SingI d => TFS d -> TFS d
tfs_sin t = unsafePerformIO $ apply0_ tSin t
  where
    tSin t' = apply0Tensor c_THFloatTensor_sin t'
{-# NOINLINE tfs_sin #-}

tfs_asin :: SingI d => TFS d -> TFS d
tfs_asin t = unsafePerformIO $ apply0_ tAsin t
  where
    tAsin t' = apply0Tensor c_THFloatTensor_asin t'
{-# NOINLINE tfs_asin #-}

tfs_sinh :: SingI d => TFS d -> TFS d
tfs_sinh t = unsafePerformIO $ apply0_ tSinh t
  where
    tSinh t' = apply0Tensor c_THFloatTensor_sinh t'
{-# NOINLINE tfs_sinh #-}

tfs_tan :: SingI d => TFS d -> TFS d
tfs_tan t = unsafePerformIO $ apply0_ tTan t
  where
    tTan t' = apply0Tensor c_THFloatTensor_tan t'
{-# NOINLINE tfs_tan #-}

tfs_atan :: SingI d => TFS d -> TFS d
tfs_atan t = unsafePerformIO $ apply0_ tAtan t
  where
    tAtan t' = apply0Tensor c_THFloatTensor_atan t'
{-# NOINLINE tfs_atan #-}

tfs_tanh :: SingI d => TFS d -> TFS d
tfs_tanh t = unsafePerformIO $ apply0_ tTanh t
  where
    tTanh t' = apply0Tensor c_THFloatTensor_tanh t'
{-# NOINLINE tfs_tanh #-}

tfs_pow :: SingI d => TFS d -> Float -> TFS d
tfs_pow t value = unsafePerformIO $ do
  let res = tfs_new
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor res)
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    liftIO $ c_THFloatTensor_pow rPtr tPtr valueC
  pure res
  where
    valueC = realToFrac value
{-# NOINLINE tfs_pow #-}

tfs_tpow :: SingI d => Float -> TFS d -> TFS d
tfs_tpow value t = unsafePerformIO $ do
  let res = tfs_new
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor res)
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    liftIO $ c_THFloatTensor_tpow rPtr valueC tPtr
  pure res
  where
    valueC = realToFrac value
{-# NOINLINE tfs_tpow #-}

tfs_sqrt :: SingI d => TFS d -> TFS d
tfs_sqrt t = unsafePerformIO $ apply0_ tSqrt t
  where
    tSqrt t' = apply0Tensor c_THFloatTensor_sqrt t'
{-# NOINLINE tfs_sqrt #-}

tfs_rsqrt :: SingI d => TFS d -> TFS d
tfs_rsqrt t = unsafePerformIO $ apply0_ tRsqrt t
  where
    tRsqrt t' = apply0Tensor c_THFloatTensor_rsqrt t'
{-# NOINLINE tfs_rsqrt #-}

tfs_ceil :: SingI d => TFS d -> TFS d
tfs_ceil t = unsafePerformIO $ apply0_ tCeil t
  where
    tCeil t' = apply0Tensor c_THFloatTensor_ceil t'
{-# NOINLINE tfs_ceil #-}

tfs_floor :: SingI d => TFS d -> TFS d
tfs_floor t = unsafePerformIO $ apply0_ tFloor t
  where
    tFloor t' = apply0Tensor c_THFloatTensor_floor t'
{-# NOINLINE tfs_floor #-}

tfs_round :: SingI d => TFS d -> TFS d
tfs_round t = unsafePerformIO $ apply0_ tRound t
  where
    tRound t' = apply0Tensor c_THFloatTensor_round t'
{-# NOINLINE tfs_round #-}


-- ----------------------------------------
-- c* cadd, cmul, cdiv, cpow, ...
-- ----------------------------------------

-- usually this is 1 mutation arg + 2 parameter args
type Raw3Arg = Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- usually this is 1 mutation arg + 3 parameter args
type Raw4Arg = Ptr CTHFloatTensor -> Ptr CTHFloatTensor ->Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

apply2 :: (SingI d1, SingI d2, SingI d3) =>
  Raw3Arg -> (TFS d1) -> (TFS d2) -> IO (TFS d3)
apply2 fun t src = do
  let r_ = tfs_new
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor r_)
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    srcPtr <- managed $ withForeignPtr (tdsTensor src)
    liftIO $ fun rPtr tPtr srcPtr
  pure r_

apply3 :: (SingI d1, SingI d2, SingI d3, SingI d4) =>
  Raw4Arg -> (TFS d1) -> (TFS d2) -> (TFS d3) -> IO (TFS d4)
apply3 fun t src1 src2 = do
  let r_ = tfs_new
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor r_)
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    src1Ptr <- managed $ withForeignPtr (tdsTensor src1)
    src2Ptr <- managed $ withForeignPtr (tdsTensor src2)
    liftIO $ fun rPtr tPtr src1Ptr src2Ptr
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
tfs_cadd :: SingI d => (TFS d) -> Float -> (TFS d) -> (TFS d)
tfs_cadd t scale src = unsafePerformIO $
  apply2 ((swap1 c_THFloatTensor_cadd) scaleC) t src
  where scaleC = realToFrac scale
{-# NOINLINE tfs_cadd #-}

tfs_csub :: SingI d => (TFS d) -> Float -> (TFS d) -> (TFS d)
tfs_csub t scale src = unsafePerformIO $ do
  apply2 ((swap1 c_THFloatTensor_csub) scaleC) t src
  where scaleC = realToFrac scale
{-# NOINLINE tfs_csub #-}

tfs_cmul :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cmul t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cmul t src
{-# NOINLINE tfs_cmul #-}

tfs_square :: SingI d => TFS d -> TFS d
tfs_square t = tfs_cmul t t

tfs_cpow :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cpow t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cpow t src
{-# NOINLINE tfs_cpow #-}

tfs_cdiv :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cdiv t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cdiv t src
{-# NOINLINE tfs_cdiv #-}

tfs_clshift :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_clshift t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_clshift t src
{-# NOINLINE tfs_clshift #-}

tfs_crshift :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_crshift t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_crshift t src
{-# NOINLINE tfs_crshift #-}

tfs_cfmod :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cfmod t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cfmod t src
{-# NOINLINE tfs_cfmod #-}

tfs_cremainder :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cremainder t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cremainder t src
{-# NOINLINE tfs_cremainder #-}

tfs_cbitand :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cbitand  t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cbitand t src
{-# NOINLINE tfs_cbitand #-}

tfs_cbitor :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cbitor  t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cbitor t src
{-# NOINLINE tfs_cbitor #-}

tfs_cbitxor :: SingI d => (TFS d) -> (TFS d) -> (TFS d)
tfs_cbitxor t src = unsafePerformIO $ do
  apply2 c_THFloatTensor_cbitxor t src
{-# NOINLINE tfs_cbitxor #-}

-- TODO - fix constraints on type signatures for addcmul and addcdiv

-- tfs_addcmul :: SingI d => (TFS d) -> Float -> (TFS d) -> (TFS d) -> (TFS d)
-- tfs_addcmul t scale src1 src2 = unsafePerformIO $ do
--   apply3 ((swap2 c_THFloatTensor_addcmul) scaleC) t src1 src2
--   where scaleC = (realToFrac scale) :: CFloat

-- tfs_addcdiv :: SingI d => (TFS d) -> Float -> (TFS d) -> (TFS d) -> (TFS d)
-- tfs_addcdiv t scale src1 src2 = unsafePerformIO $ do
--   apply3 ((swap2 c_THFloatTensor_addcdiv) scaleC) t src1 src2
--   where scaleC = (realToFrac scale) :: CFloat

-- |beta * t + alpha * (src1 #> src2)
tfs_addmv :: (KnownNat c, KnownNat r) =>
  Float -> (TFS '[r]) -> Float -> (TFS '[r, c]) -> (TFS '[c]) -> (TFS '[r])
tfs_addmv beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THFloatTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CFloat, CFloat)
{-# NOINLINE tfs_addmv #-}

-- | added simplified use of addmv: src1 #> src2
tfs_mv :: (KnownNat c, KnownNat r) => (TFS '[r, c]) -> (TFS '[c]) -> (TFS '[r])
tfs_mv m v = tfs_addmv 0.0 tfs_new 1.0 m v

apply1 fun t = do
  let r_ = tfs_new
  withForeignPtr (tdsTensor r_)
    (\rPtr ->
       withForeignPtr (tdsTensor t)
         (\tPtr ->
            fun rPtr tPtr
         )
    )
  pure r_

type Ret2Fun =
  Ptr CTHFloatTensor -> Ptr CTHLongTensor -> Ptr CTHFloatTensor -> CInt -> CInt -> IO ()

ret2 :: SingI d => Ret2Fun -> (TFS d) -> Int -> Bool -> IO ((TFS d), TensorLong)
ret2 fun t dimension keepdim = do
  let values_ = tfs_new
  let indices_ = tl_new (tfs_dim t)
  runManaged $ do
    vPtr <- managed $ withForeignPtr (tdsTensor values_)
    iPtr <- managed $ withForeignPtr (tlTensor indices_)
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    liftIO $ fun vPtr iPtr tPtr dimensionC keepdimC
  pure (values_, indices_)
  where
    keepdimC = if keepdim then 1 else 0
    dimensionC = fromIntegral dimension

tfs_addmm :: (KnownNat a, KnownNat b, KnownNat c) =>
  Float -> TFS [a,c] -> Float -> TFS [a, b] -> TFS [b, c] -> TFS [a,c]
tfs_addmm beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THFloatTensor_addmm) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CFloat, CFloat)
{-# NOINLINE tfs_addmm #-}

tfs_addr :: (KnownNat r, KnownNat c) =>
  Float -> TFS '[r, c]-> Float -> TFS '[r] -> TFS '[c]-> TFS '[r, c]
tfs_addr beta t alpha vec1 vec2 = unsafePerformIO $ do
  let r_ = tfs_new
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor r_)
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    vec1Ptr <- managed $ withForeignPtr (tdsTensor vec1)
    vec2Ptr <- managed $ withForeignPtr (tdsTensor vec2)
    liftIO $ c_THFloatTensor_addr rPtr betaC tPtr alphaC vec1Ptr vec2Ptr
  pure r_
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CFloat, CFloat)
{-# NOINLINE tfs_addr #-}

tfs_outer :: (KnownNat r, KnownNat c) =>
             TFS '[r] -> TFS '[c] -> TFS '[r, c]
tfs_outer vec1 vec2 = tfs_addr 0.0 tfs_new 1.0 vec1 vec2

-- TODO- add proper type signature with dimensions specified
-- tfs_addbmm :: Float -> (TFS d) -> Float -> (TFS d) -> (TFS d) -> (TFS d)
tfs_addbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THFloatTensor_addbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CFloat, CFloat)
{-# NOINLINE tfs_addbmm #-}

-- TODO- add proper type signature with dimensions specified
-- tfs_baddbmm :: Float -> (TFS d) -> Float -> (TFS d) -> (TFS d) -> (TFS d)
tfs_baddbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THFloatTensor_baddbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CFloat, CFloat)
{-# NOINLINE tfs_baddbmm #-}

-- tfs_match :: (TFS d) -> (TFS d) -> Float -> (TFS d)
tfs_match m1 m2 gain = unsafePerformIO $ do
  apply2 ((swap c_THFloatTensor_match) gainC) m1 m2
  where
    gainC = realToFrac gain
    swap fun gain b c d = fun b c d gain
{-# NOINLINE tfs_match #-}

tfs_numel :: (TFS d) -> Int
tfs_numel t = unsafePerformIO $ do
  result <- apply0_ c_THFloatTensor_numel t
  pure $ fromIntegral result
{-# NOINLINE tfs_numel #-}

{-
TODO : need type computations for resulting dimensions
-}

-- TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tfs_max :: SingI d => (TFS d) -> Int -> Bool -> ((TFS d), TensorLong)
tfs_max t dimension keepdim = unsafePerformIO $
  ret2 c_THFloatTensor_max t dimension keepdim
{-# NOINLINE tfs_max #-}

-- TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
-- tfs_minT :: (TFS d) -> Int -> Bool -> ((TFS d), TensorLong)
tfs_min t dimension keepdim = unsafePerformIO $
  ret2 c_THFloatTensor_min t dimension keepdim
{-# NOINLINE tfs_min #-}

-- TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
-- tfs_kthvalue :: (TFS d) -> Int -> Int -> Bool -> ((TFS d), TensorLong)
tfs_kthvalue t k dimension keepdim = unsafePerformIO $
  ret2 ((swap c_THFloatTensor_kthvalue) kC) t dimension keepdim
  where
    swap fun a b c d e f = fun b c d a e f -- curry k (4th argument)
    kC = fromIntegral k
{-# NOINLINE tfs_kthvalue #-}

-- TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
-- tfs_mode :: (TFS d) -> Int -> Bool -> ((TFS d), TensorLong)
tfs_mode t dimension keepdim = unsafePerformIO $
  ret2 c_THFloatTensor_mode t dimension keepdim
{-# NOINLINE tfs_mode #-}

-- TH_API void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
tfs_median :: SingI d => (TFS d) -> Int -> Bool -> ((TFS d), TensorLong)
tfs_median t dimension keepdim = unsafePerformIO $
  ret2 c_THFloatTensor_median t dimension keepdim
{-# NOINLINE tfs_median #-}

-- TODO - types
-- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
-- tfs_sum :: (TFS d) -> Int -> Bool -> (TFS d)
tfs_sum t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THFloatTensor_sum) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0
{-# NOINLINE tfs_sum #-}

-- |row sums of a matrix
tfs_rowsum :: (KnownNat r, KnownNat c) => TFS [r, c] -> TFS [1, c]
tfs_rowsum t = tfs_sum t 0 True

-- |column sums of a matrix
tfs_colsum :: (KnownNat r, KnownNat c) => TFS [r, c] -> TFS [r, 1]
tfs_colsum t = tfs_sum t 1 True

-- TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
-- tfs_prod :: (TFS d) -> Int -> Bool -> (TFS d)
tfs_prod t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THFloatTensor_prod) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0
{-# NOINLINE tfs_prod #-}

-- TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
-- tfs_cumsum :: (TFS d) -> Int -> (TFS d)
tfs_cumsum t dimension = unsafePerformIO $ do
  apply1 ((swap c_THFloatTensor_cumsum) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension
{-# NOINLINE tfs_cumsum #-}

-- TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
-- tfs_cumprod :: (TFS d) -> Int -> (TFS d)
tfs_cumprod t dimension = unsafePerformIO $ do
  apply1 ((swap c_THFloatTensor_cumprod) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension
{-# NOINLINE tfs_cumprod #-}

-- TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);
tfs_sign :: SingI d => (TFS d) -> (TFS d)
tfs_sign t = unsafePerformIO $ do
  apply1 c_THFloatTensor_sign t
{-# NOINLINE tfs_sign #-}

-- TH_API accreal THTensor_(trace)(THTensor *t);
tfs_trace :: SingI d => (TFS d) -> Float
tfs_trace t = realToFrac $ unsafePerformIO $ do
  apply0_ c_THFloatTensor_trace t
{-# NOINLINE tfs_trace #-}

-- TH_API void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
tfs_cross a b dimension = unsafePerformIO $ do
  apply2 ((swap c_THFloatTensor_cross) dimensionC) a b
  where
    dimensionC = fromIntegral dimension
    swap fun a b c d = fun b c d a
{-# NOINLINE tfs_cross #-}

-- TH_API void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
tfs_cmax t src = unsafePerformIO $ apply2 c_THFloatTensor_cmax t src
{-# NOINLINE tfs_cmax #-}

-- TH_API void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
tfs_cmin t src = unsafePerformIO $ apply2 c_THFloatTensor_cmin t src
{-# NOINLINE tfs_cmin #-}

----------

-- |Test for equality between all elements of two tensors between two tensors
tfs_equal :: SingI d => (TFS d) -> (TFS d) -> Bool
tfs_equal ta tb = unsafePerformIO $ do
  res <- fromIntegral <$> withForeignPtr (tdsTensor ta)
         (\taPtr ->
             withForeignPtr (tdsTensor tb)
               (\tbPtr ->
                   pure $ c_THFloatTensor_equal taPtr tbPtr
               )
         )
  pure $ res == 1
{-# NOINLINE tfs_equal #-}

-- tfs_geValue :: SingI d => (TFS d) -> (TFS d) -> Float -> Bool
-- tfs_geValue ta tb = unsafePerformIO $ do
--   let res = tbs_new
--   res <- fromIntegral <$> withForeignPtr (tdsTensor ta)
--          (\taPtr ->
--              withForeignPtr (tdsTensor tb)
--                (\tbPtr ->
--                    pure $ c_THFloatTensor_geValue taPtr tbPtr
--                )
--          )
--   pure $ res == 1
-- {-# NOINLINE tfs_geValue #-}

-- |Concatenate two vectors
tfs_cat :: forall n1 n2 n . (SingI n1, SingI n2, SingI n, n ~ Sum [n1, n2]) =>
  TFS '[n1] -> TFS '[n2] -> TFS '[n]
tfs_cat ta tb = unsafePerformIO $ do
  let r_ = tfs_new :: TFS '[n]
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor r_)
    taPtr <- managed $ withForeignPtr (tdsTensor ta)
    tbPtr <- managed $ withForeignPtr (tdsTensor tb)
    liftIO $ c_THFloatTensor_cat rPtr taPtr tbPtr 0
  pure r_
{-# NOINLINE tfs_cat #-}

-- |Create a diagonal matrix from a 1D vector
tfs_diag :: forall d . SingI d => TFS '[d] -> TFS '[d,d]
tfs_diag t = unsafePerformIO $ do
  let r_ = tfs_new :: TFS '[d,d]
  runManaged $ do
    rPtr <- managed $ withForeignPtr (tdsTensor r_)
    tPtr <- managed $ withForeignPtr (tdsTensor t)
    liftIO $ c_THFloatTensor_diag rPtr tPtr k
  pure r_
  where k = 0
{-# NOINLINE tfs_diag #-}
-}
