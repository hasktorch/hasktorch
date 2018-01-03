{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE InstanceSigs, RankNTypes, PolyKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Core.Tensor.Dynamic.Generic
  ( genericWrapRaw
  , genericP
  , genericToList
  , genericFromListNd
  , genericFromList1d
  , genericResize
  , genericGet
  , genericNew
  , genericNew'
  , genericDynamicDims
  ) where

import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Dim (Dim(..), SomeDims(..), someDimsM)
import Torch.Core.Tensor.Generic.Internal (CTHDoubleTensor, CTHFloatTensor, HaskReal)
import qualified THDoubleTensor as T
import qualified THLongTensor as T
import qualified Torch.Core.Tensor.Generic as Gen
import qualified Torch.Core.Tensor.Dim as Dim

import Torch.Core.Tensor.Dynamic.Generic.Internal

instance IsList TensorDouble where
  type Item TensorDouble = Double
  fromList = genericFromList1d realToFrac
  toList = genericToList realToFrac

instance IsList TensorFloat where
  type Item TensorFloat = Float
  fromList = genericFromList1d realToFrac
  toList = genericToList realToFrac

genericWrapRaw :: GenericOps' t => Ptr (TensorPtrType t) -> IO t
genericWrapRaw tensor = construct <$> newForeignPtr Gen.p_free tensor

genericP :: (GenericOps' t, Show (HaskReal' t)) => t -> IO ()
genericP t = withForeignPtr (getForeign t) Gen.dispRaw

genericToList :: GenericOps' t => (HaskReal' t -> Item t) -> t -> [Item t]
genericToList translate t = unsafePerformIO $ withForeignPtr (getForeign t) (pure . map translate . Gen.flatten)
{-# NOINLINE genericToList #-}

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME(stites): This should go in MonadThrow
genericFromListNd
  :: (GenericMath' t, Num (HaskReal' t), k ~ Nat)
  => (Item t -> HaskReal' t) -> Dim (d::[k]) -> [Item t] -> t
genericFromListNd translate d l =
  if fromIntegral (product (Dim.dimVals d)) == length l
  then genericResize (genericFromList1d translate l) d
  else error "Incorrect tensor dimensions specified."

-- | Initialize a 1D tensor from a list
genericFromList1d
  :: forall t k . (GenericMath' t, Num (HaskReal' t), k ~ Nat)
  => (Item t -> HaskReal' t) -> [Item t] -> t
genericFromList1d translate l = unsafePerformIO $ do
  sdims <- someDimsM [length l]
  let res = genericNew' sdims
  mapM_ (mutTensor (getForeign res)) (zip [0..length l - 1] l)
  pure res
 where
  mutTensor :: ForeignPtr (TensorPtrType t) -> (Int, Item t) -> IO ()
  mutTensor t (idx, value) = withForeignPtr t $ \tp -> Gen.c_set1d tp (fromIntegral idx) (translate value)
{-# NOINLINE genericFromList1d #-}

-- |Copy contents of tensor into a new one of specified size
genericResize
  :: forall t d . (GenericMath' t, Num (HaskReal' t), k ~ Nat)
  => t -> Dim (d::[k]) -> t
genericResize t d = unsafePerformIO $ do
  let resDummy = genericNew d :: t
  newPtr <- withForeignPtr (getForeign t) Gen.c_newClone
  newFPtr <- newForeignPtr Gen.p_free newPtr
  with2ForeignPtrs newFPtr (getForeign resDummy) Gen.c_resizeAs
  pure $ construct newFPtr
{-# NOINLINE genericResize #-}

with2ForeignPtrs :: ForeignPtr f -> ForeignPtr f -> (Ptr f -> Ptr f -> IO x) -> IO x
with2ForeignPtrs fp1 fp2 fn = withForeignPtr fp1 (withForeignPtr fp2 . fn)

genericGet :: (GenericOps' t, k ~ Nat) => (HaskReal (TensorPtrType t) -> Item t) -> Dim (d::[k]) -> t -> Item t
genericGet translate loc tensor = unsafePerformIO $ withForeignPtr
  (getForeign tensor)
  (\t -> pure . translate $ t `Gen.genericGet` loc)
{-# NOINLINE genericGet #-}

genericNewWithTensor :: (GenericOps' t) => t -> t
genericNewWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) Gen.c_newWithTensor
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericNewWithTensor #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
genericNew
  :: (GenericMath' t, Num (HaskReal' t), k ~ Nat)
  => Dim (d::[k]) -> t
genericNew dims = unsafePerformIO $ do
  newPtr <- Gen.constant dims 0
  fPtr <- newForeignPtr Gen.p_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ construct fPtr
{-# NOINLINE genericNew #-}

genericNew'
  :: (GenericMath' t, Num (HaskReal' t), k ~ Nat)
  => SomeDims -> t
genericNew' (SomeDims ds) = genericNew ds

genericFree_ :: THTensor t => t -> IO ()
genericFree_ t = finalizeForeignPtr $! getForeign t

genericInit :: (GenericMath' t, k ~ Nat) => (Item t -> HaskReal' t) -> Dim (d::[k]) -> Item t -> t
genericInit translate ds val = unsafePerformIO $ do
  newPtr <- Gen.constant ds (translate val)
  fPtr <- newForeignPtr Gen.p_free newPtr
  withForeignPtr fPtr (Gen.inplaceFill translate val)
  pure $ construct fPtr
{-# NOINLINE genericInit #-}

genericTranspose :: GenericOps' t => Word -> Word -> t -> t
genericTranspose (fromIntegral->dim1C) (fromIntegral->dim2C) t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p dim1C dim2C)
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericTranspose #-}

genericTrans :: GenericOps' t => t -> t
genericTrans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p 1 0)
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericTrans #-}

genericDynamicDims :: GenericOps' t => t -> SomeDims
genericDynamicDims t = unsafePerformIO $
  withForeignPtr (getForeign t) (pure . Gen.getDynamicDim)
{-# NOINLINE genericDynamicDims #-}


{-
-------------------------------------------------------------------------------
-- DOUBLE MATH
-------------------------------------------------------------------------------

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
{-# NOINLINE apply1_ #-}

-- | Generalize non-mutating collapse of a tensor to a constant or another tensor
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
{-# NOINLINE apply0Tensor #-}

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
       withForeignPtr (getForeign indices_)
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
{-# NOINLINE td_fill #-}

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
{-# NOINLINE td_dot #-}

-- ----------------------------------------
-- Collapse to constant operations
-- ----------------------------------------

td_minAll :: TensorDouble -> Double
td_minAll tensor = unsafePerformIO $ apply0_ tMinAll tensor
  where
    tMinAll t = realToFrac $ c_THDoubleTensor_minall t
{-# NOINLINE td_minAll #-}

td_maxAll :: TensorDouble -> Double
td_maxAll tensor = unsafePerformIO $ apply0_ tMaxAll tensor
  where
    tMaxAll t = realToFrac $ c_THDoubleTensor_maxall t
{-# NOINLINE td_maxAll #-}

td_medianAll :: TensorDouble -> Double
td_medianAll tensor = unsafePerformIO $ apply0_ tMedianAll tensor
  where
    tMedianAll t = realToFrac $ c_THDoubleTensor_medianall t
{-# NOINLINE td_medianAll #-}

td_sumAll :: TensorDouble -> Double
td_sumAll tensor = unsafePerformIO $ apply0_ tSumAll tensor
  where
    tSumAll t = realToFrac $ c_THDoubleTensor_sumall t
{-# NOINLINE td_sumAll #-}

td_prodAll :: TensorDouble -> Double
td_prodAll tensor = unsafePerformIO $ apply0_ tProdAll tensor
  where
    tProdAll t = realToFrac $ c_THDoubleTensor_prodall t
{-# NOINLINE td_prodAll #-}

td_meanAll :: TensorDouble -> Double
td_meanAll tensor = unsafePerformIO $ apply0_ tMeanAll tensor
  where
    tMeanAll t = realToFrac $ c_THDoubleTensor_meanall t
{-# NOINLINE td_meanAll #-}

-- ----------------------------------------
-- Tensor to Tensor transformation
-- ----------------------------------------

td_neg :: TensorDouble -> TensorDouble
td_neg tensor = unsafePerformIO $ apply0_ tNeg tensor
  where
    tNeg t = apply0Tensor c_THDoubleTensor_neg (tdDim tensor) t
{-# NOINLINE td_neg #-}

td_cinv :: TensorDouble -> TensorDouble
td_cinv tensor = unsafePerformIO $ apply0_ cinv tensor
  where
    cinv t = apply0Tensor c_THDoubleTensor_cinv (tdDim tensor) t
{-# NOINLINE td_cinv #-}

td_abs :: TensorDouble -> TensorDouble
td_abs tensor = unsafePerformIO $ apply0_ tAbs tensor
  where
    tAbs t = apply0Tensor c_THDoubleTensor_abs (tdDim tensor) t
{-# NOINLINE td_abs #-}

td_sigmoid :: TensorDouble -> TensorDouble
td_sigmoid tensor = unsafePerformIO $ apply0_ tSigmoid tensor
  where
    tSigmoid t = apply0Tensor c_THDoubleTensor_sigmoid (tdDim tensor) t
{-# NOINLINE td_sigmoid #-}

td_log :: TensorDouble -> TensorDouble
td_log tensor = unsafePerformIO $ apply0_ tLog tensor
  where
    tLog t = apply0Tensor c_THDoubleTensor_log (tdDim tensor) t
{-# NOINLINE td_log #-}

td_lgamma :: TensorDouble -> TensorDouble
td_lgamma tensor = unsafePerformIO $ apply0_ tLgamma tensor
  where
    tLgamma t = apply0Tensor c_THDoubleTensor_lgamma (tdDim tensor) t
{-# NOINLINE td_lgamma #-}

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
  let r_ = td_new (tdDim t)
  withForeignPtr (tdTensor r_)
    (\rPtr ->
       withForeignPtr (tdTensor t)
         (\tPtr ->
            withForeignPtr (tdTensor src)
              (\srcPtr ->
                 c_THDoubleTensor_cadd rPtr tPtr scaleC srcPtr
              )
         )
    )
  pure r_
  where scaleC = realToFrac scale
{-# NOINLINE td_cadd #-}

td_csub :: TensorDouble -> Double -> TensorDouble -> TensorDouble
td_csub t scale src = unsafePerformIO $ do
  checkdim t src "csub"
  let r_ = td_new (tdDim t)
  withForeignPtr (tdTensor r_)
    (\rPtr ->
       withForeignPtr (tdTensor t)
         (\tPtr ->
            withForeignPtr (tdTensor src)
              (\srcPtr ->
                 c_THDoubleTensor_csub rPtr tPtr scaleC srcPtr
              )
         )
    )
  pure r_
  where scaleC = realToFrac scale
{-# NOINLINE td_csub #-}

td_cmul :: TensorDouble -> TensorDouble -> TensorDouble
td_cmul t src = unsafePerformIO $ do
  checkdim t src "cmul"
  apply2 c_THDoubleTensor_cmul t src
{-# NOINLINE td_cmul #-}

td_cpow :: TensorDouble -> TensorDouble -> TensorDouble
td_cpow t src = unsafePerformIO $ do
  checkdim t src "cpow"
  apply2 c_THDoubleTensor_cpow t src
{-# NOINLINE td_cpow #-}

td_cdiv :: TensorDouble -> TensorDouble -> TensorDouble
td_cdiv t src = unsafePerformIO $ do
  checkdim t src "cdiv"
  apply2 c_THDoubleTensor_cdiv t src
{-# NOINLINE td_cdiv #-}

td_clshift :: TensorDouble -> TensorDouble -> TensorDouble
td_clshift t src = unsafePerformIO $ do
  checkdim t src "clshift"
  apply2 c_THDoubleTensor_clshift t src
{-# NOINLINE td_clshift #-}

td_crshift :: TensorDouble -> TensorDouble -> TensorDouble
td_crshift t src = unsafePerformIO $ do
  checkdim t src "crshift"
  apply2 c_THDoubleTensor_crshift t src
{-# NOINLINE td_crshift #-}

td_cfmod :: TensorDouble -> TensorDouble -> TensorDouble
td_cfmod t src = unsafePerformIO $ do
  checkdim t src "cfmod"
  apply2 c_THDoubleTensor_cfmod t src
{-# NOINLINE td_cfmod #-}

td_cremainder :: TensorDouble -> TensorDouble -> TensorDouble
td_cremainder t src = unsafePerformIO $ do
  checkdim t src "cremainder"
  apply2 c_THDoubleTensor_cremainder t src
{-# NOINLINE td_cremainder #-}

td_cbitand :: TensorDouble -> TensorDouble -> TensorDouble
td_cbitand  t src = unsafePerformIO $ do
  checkdim t src "cbitand"
  apply2 c_THDoubleTensor_cbitand t src
{-# NOINLINE td_cbitand #-}

td_cbitor :: TensorDouble -> TensorDouble -> TensorDouble
td_cbitor  t src = unsafePerformIO $ do
  apply2 c_THDoubleTensor_cbitor t src
{-# NOINLINE td_cbitor #-}

td_cbitxor :: TensorDouble -> TensorDouble -> TensorDouble
td_cbitxor t src = unsafePerformIO $ do
  checkdim t src "cbitxor"
  apply2 c_THDoubleTensor_cbitxor t src
{-# NOINLINE td_cbitxor #-}

td_addcmul :: TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addcmul t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcmul) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble
{-# NOINLINE td_addcmul #-}

td_addcdiv :: TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addcdiv t scale src1 src2 = unsafePerformIO $ do
  apply3 ((swap2 c_THDoubleTensor_addcdiv) scaleC) t src1 src2
  where scaleC = (realToFrac scale) :: CDouble
{-# NOINLINE td_addcdiv #-}

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
{-# NOINLINE td_addmv #-}

-- |No dimension checks (halts execution if they're incorrect)
td_addmv_fast :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addmv_fast beta t alpha src1 src2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addmv) betaC alphaC) t src1 src2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)
{-# NOINLINE td_addmv_fast #-}

td_mv :: TensorDouble -> TensorDouble -> TensorDouble
td_mv mat vec =
  td_addmv 0.0 zero 1.0 mat vec
  where
    zero = td_new $ (D1 ((^. _1) . d2 . tdDim $ mat))

-- |No dimension checks (halts execution if they're incorrect)
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
{-# NOINLINE td_addmm #-}

-- |outer product - see https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchaddrres-v1-mat-v2-vec1-vec2
td_addr :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addr beta t alpha vec1 vec2 = unsafePerformIO $ do
  case (dimt, dim1, dim2) of
    (D2 _, D1 _, D1 _) -> pure ()
    _ -> error "Expected: 1D vector + 2D matrix x 1D vector"
  unless (d2 dimt ^. _1 == d1 dim1) $ error "Matrix dimension mismatch with vec1"
  unless (d2 dimt ^. _2 == d1 dim2) $ error "Matrix dimension mismatch with vec2"
  let r_ = td_new (tdDim t)
  withForeignPtr (tdTensor r_)
    (\rPtr ->
       withForeignPtr (tdTensor t)
         (\tPtr ->
            withForeignPtr (tdTensor vec1)
              (\vec1Ptr ->
                 withForeignPtr (tdTensor vec2)
                   (\vec2Ptr ->
                      c_THDoubleTensor_addr rPtr betaC tPtr alphaC vec1Ptr vec2Ptr
                   )
              )
         )
    )
  pure r_
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)
    (dim1, dim2, dimt) = (tdDim vec1, tdDim vec2, tdDim t)
{-# NOINLINE td_addr #-}

td_outer vec1 vec2 = unsafePerformIO $ do
  case (dim1, dim2) of
    (D1 _, D1 _) -> pure ()
    _ -> error "Expected: 1D vectors"
  pure $ td_addr 0.0 emptyMtx 1.0 vec1 vec2
  where
    (dim1, dim2) = (tdDim vec1, tdDim vec2)
    emptyMtx = td_init (D2 ((d1 dim1), (d1 dim2))) 0.0
{-# NOINLINE td_outer #-}


td_addbmm :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_addbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_addbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)
{-# NOINLINE td_addbmm #-}

td_baddbmm :: Double -> TensorDouble -> Double -> TensorDouble -> TensorDouble -> TensorDouble
td_baddbmm beta t alpha batch1 batch2 = unsafePerformIO $ do
  apply3 ((swap3 c_THDoubleTensor_baddbmm) betaC alphaC) t batch1 batch2
  where
    (betaC, alphaC) = (realToFrac beta, realToFrac alpha) :: (CDouble, CDouble)
{-# NOINLINE td_baddbmm #-}

td_match :: TensorDouble -> TensorDouble -> Double -> TensorDouble
td_match m1 m2 gain = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_match) gainC) m1 m2
  where
    gainC = realToFrac gain
    swap fun gain b c d = fun b c d gain
{-# NOINLINE td_match #-}

td_numel :: TensorDouble -> Int
td_numel t = unsafePerformIO $ do
  result <- apply0_ c_THDoubleTensor_numel t
  pure $ fromIntegral result
{-# NOINLINE td_numel #-}

-- TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_max :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_max t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_max t dimension keepdim
{-# NOINLINE td_max #-}

-- TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_min :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_min t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_min t dimension keepdim
{-# NOINLINE td_min #-}

-- TH_API void THTensor_(kthvalue)(THTensor *values_, THLongTensor *indices_, THTensor *t, long k, int dimension, int keepdim);
td_kthvalue :: TensorDouble -> Int -> Int -> Bool -> (TensorDouble, TensorLong)
td_kthvalue t k dimension keepdim = unsafePerformIO $
  ret2 ((swap c_THDoubleTensor_kthvalue) kC) t dimension keepdim
  where
    swap fun a b c d e f = fun b c d a e f -- curry k (4th argument)
    kC = fromIntegral k
{-# NOINLINE td_kthvalue #-}

-- TH_API void THTensor_(mode)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_mode :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_mode t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_mode t dimension keepdim
{-# NOINLINE td_mode #-}

-- TH_API void THTensor_(median)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
td_median :: TensorDouble -> Int -> Bool -> (TensorDouble, TensorLong)
td_median t dimension keepdim = unsafePerformIO $
  ret2 c_THDoubleTensor_median t dimension keepdim
{-# NOINLINE td_median #-}

-- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
td_sum :: TensorDouble -> Int -> Bool -> TensorDouble
td_sum t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_sum) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0
{-# NOINLINE td_sum #-}

-- TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
td_prod :: TensorDouble -> Int -> Bool -> TensorDouble
td_prod t dimension keepdim = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_prod) dimensionC keepdimC) t
  where
    swap fun a b c d = fun c d a b
    dimensionC = fromIntegral dimension
    keepdimC = if keepdim then 1 else 0
{-# NOINLINE td_prod #-}

-- TH_API void THTensor_(cumsum)(THTensor *r_, THTensor *t, int dimension);
td_cumsum :: TensorDouble -> Int -> TensorDouble
td_cumsum t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumsum) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension
{-# NOINLINE td_cumsum #-}

-- TH_API void THTensor_(cumprod)(THTensor *r_, THTensor *t, int dimension);
td_cumprod :: TensorDouble -> Int -> TensorDouble
td_cumprod t dimension = unsafePerformIO $ do
  apply1 ((swap c_THDoubleTensor_cumprod) dimensionC) t
  where
    swap fun a b c = fun b c a
    dimensionC = fromIntegral dimension
{-# NOINLINE td_cumprod #-}

-- TH_API void THTensor_(sign)(THTensor *r_, THTensor *t);
td_sign :: TensorDouble -> TensorDouble
td_sign t = unsafePerformIO $ do
  apply1 c_THDoubleTensor_sign t
{-# NOINLINE td_sign #-}

-- TH_API accreal THTensor_(trace)(THTensor *t);
td_trace :: TensorDouble -> Double
td_trace t = realToFrac $ unsafePerformIO $ do
  apply0_ c_THDoubleTensor_trace t
{-# NOINLINE td_trace #-}

-- TH_API void THTensor_(cross)(THTensor *r_, THTensor *a, THTensor *b, int dimension);
td_cross :: TensorDouble -> TensorDouble -> Int -> TensorDouble
td_cross a b dimension = unsafePerformIO $ do
  apply2 ((swap c_THDoubleTensor_cross) dimensionC) a b
  where
    dimensionC = fromIntegral dimension
    swap fun a b c d = fun b c d a
{-# NOINLINE td_cross #-}

-- TH_API void THTensor_(cmax)(THTensor *r, THTensor *t, THTensor *src);
td_cmax :: TensorDouble -> TensorDouble -> TensorDouble
td_cmax t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmax t src
{-# NOINLINE td_cmax #-}

-- TH_API void THTensor_(cmin)(THTensor *r, THTensor *t, THTensor *src);
td_cmin :: TensorDouble -> TensorDouble -> TensorDouble
td_cmin t src = unsafePerformIO $ apply2 c_THDoubleTensor_cmin t src
{-# NOINLINE td_cmin #-}

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
{-# NOINLINE td_eye #-}

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
{-# NOINLINE td_equal #-}

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
{-# NOINLINE td_round #-}

-- -- TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);


-}
  {-
-------------------------------------------------------------------------------
-- DOUBLE RANDOM
-------------------------------------------------------------------------------

  ( td_random
  , td_clampedRandom
  , td_cappedRandom
  , td_geometric
  , td_bernoulli
  , td_bernoulliFloat
  , td_bernoulliDouble
  , td_uniform
  , td_normal
  , td_exponential
  , td_cauchy
  , td_multinomial

  , module Torch.Core.Random
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import Torch.Core.Random

import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom

import THFloatTensor

td_random :: TensorDouble -> RandGen -> IO TensorDouble
td_random self gen = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_random s g
         )
    )
  pure self

td_clampedRandom self gen minVal maxVal = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_clampedRandom s g minC maxC
         )
    )
  pure self
  where (minC, maxC) = (fromIntegral minVal, fromIntegral maxVal)

td_cappedRandom :: TensorDouble -> RandGen -> Int -> IO TensorDouble
td_cappedRandom self gen maxVal = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_cappedRandom s g maxC
         )
    )
  pure self
  where maxC = fromIntegral maxVal

-- TH_API void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
td_geometric :: TensorDouble -> RandGen -> Double -> IO TensorDouble
td_geometric self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_geometric s g pC
         )
    )
  pure self
  where pC = realToFrac p

-- TH_API void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
td_bernoulli :: TensorDouble -> RandGen -> Double -> IO TensorDouble
td_bernoulli self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_bernoulli s g pC
         )
    )
  pure self
  where pC = realToFrac p

td_bernoulliFloat :: TensorDouble -> RandGen -> TensorFloat -> IO ()
td_bernoulliFloat self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (pC)
              (\pTensor ->
                 c_THDoubleTensor_bernoulli_FloatTensor s g pTensor
              )
         )
    )
  where pC = tfTensor p

td_bernoulliDouble :: TensorDouble -> RandGen -> TensorDouble -> IO TensorDouble
td_bernoulliDouble self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (pC)
              (\pTensor ->
                 c_THDoubleTensor_bernoulli_DoubleTensor s g pTensor
              )
         )
    )
  pure self
  where pC = tdTensor p

td_uniform :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_uniform self gen a b = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_uniform s g aC bC
         )
    )
  pure self
  where aC = realToFrac a
        bC = realToFrac b

td_normal :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_normal self gen mean stdv = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_normal s g meanC stdvC
         )
    )
  pure self
  where meanC = realToFrac mean
        stdvC = realToFrac stdv

-- TH_API void THTensor_(normal_means)(THTensor *self, THGenerator *gen, THTensor *means, double stddev);
-- TH_API void THTensor_(normal_stddevs)(THTensor *self, THGenerator *gen, double mean, THTensor *stddevs);
-- TH_API void THTensor_(normal_means_stddevs)(THTensor *self, THGenerator *gen, THTensor *means, THTensor *stddevs);

td_exponential :: TensorDouble -> RandGen -> Double -> IO TensorDouble
td_exponential self gen lambda = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_exponential s g lambdaC
         )
    )
  pure self
  where lambdaC = realToFrac lambda

td_cauchy :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_cauchy self gen median sigma = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_cauchy s g medianC sigmaC
         )
    )
  pure self
  where medianC = realToFrac median
        sigmaC = realToFrac sigma

td_logNormal :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_logNormal self gen mean stdv = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_logNormal s g meanC stdvC
         )
    )
  pure self
  where meanC = realToFrac mean
        stdvC = realToFrac stdv


td_multinomial :: TensorLong -> RandGen -> TensorDouble -> Int -> Bool -> IO TensorLong
td_multinomial self gen prob_dist n_sample with_replacement = do
  withForeignPtr (getForeign self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (tdTensor prob_dist)
              (\p ->
                 c_THDoubleTensor_multinomial s g p n_sampleC with_replacementC
              )
         )
    )
  pure self
  where n_sampleC = fromIntegral n_sample
        with_replacementC = if with_replacement then 1 else 0

-- TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
-- TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THGenerator *_generator, THLongTensor *J, THTensor *q);
-- #endif

-}
