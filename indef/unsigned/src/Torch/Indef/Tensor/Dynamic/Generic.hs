{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE InstanceSigs, RankNTypes, PolyKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Indef.Tensor.Dynamic.Generic
  ( THTensorLapack'
  , THTensorMath'
  , THTensor'
  , HaskReal'

  , genericWrapRaw
  , genericP
  , genericToList
  , genericFromListNd
  , genericFromList1d
  , genericResize
  , genericGet
  , genericNew
  , genericNew'
  , genericDynamicDims

  , with2ForeignPtrs
  , with3ForeignPtrs
  , with4ForeignPtrs

  , with2THForeignRefs
  , with3THForeignRefs
  , with4THForeignRefs
  ) where

import Control.Monad (void)
import Foreign.C.Types
import Foreign (Ptr, ForeignPtr, withForeignPtr, newForeignPtr, finalizeForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Exts (fromList, toList, IsList, Item)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Indef.Tensor.Types
import Torch.Dimensions (Dim(..), SomeDims(..), someDimsM)
import qualified Torch.FFI.TH.Double.Tensor as T
import qualified Torch.FFI.TH.Long.Tensor as T
import qualified Torch.Raw.Tensor.Generic as Gen
import qualified Torch.Dimensions as Dim

import Torch.Raw.Tensor.Generic (THTensor, THTensorLapack, THTensorMath, HaskReal)

type THTensorLapack' t = (THTensorLapack (THForeignType t), THForeignRef t)
type THTensorMath' t = (THTensor (THForeignType t), THTensorMath (THForeignType t), THForeignRef t)
type THTensor' t = (THTensor (THForeignType t), THForeignRef t)
type HaskReal' t = (HaskReal (THForeignType t))


instance IsList TensorDouble where
  type Item TensorDouble = Double
  fromList = genericFromList1d realToFrac
  toList = genericToList realToFrac

genericWrapRaw :: THTensor' t => Ptr (THForeignType t) -> IO t
genericWrapRaw tensor = construct <$> newForeignPtr Gen.p_free tensor

genericP :: (THTensor' t, Show (HaskReal' t)) => t -> IO ()
genericP t = withForeignPtr (getForeign t) Gen.dispRaw

genericToList :: THTensor' t => (HaskReal' t -> Item t) -> t -> [Item t]
genericToList translate t = unsafePerformIO $ withForeignPtr (getForeign t) (pure . map translate . Gen.flatten)
{-# NOINLINE genericToList #-}

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME(stites): This should go in MonadThrow
genericFromListNd
  :: (THTensorMath' t, Num (HaskReal' t), k ~ Nat)
  => (Item t -> HaskReal' t) -> Dim (d::[k]) -> [Item t] -> t
genericFromListNd translate d l =
  if fromIntegral (product (Dim.dimVals d)) == length l
  then genericResize (genericFromList1d translate l) d
  else error "Incorrect tensor dimensions specified."

-- | Initialize a 1D tensor from a list
genericFromList1d
  :: forall t k . (THTensorMath' t, Num (HaskReal' t), k ~ Nat)
  => (Item t -> HaskReal' t) -> [Item t] -> t
genericFromList1d translate l = unsafePerformIO $ do
  sdims <- someDimsM [length l]
  let res = genericNew' sdims
  mapM_ (mutTensor (getForeign res)) (zip [0..length l - 1] l)
  pure res
 where
  mutTensor :: ForeignPtr (THForeignType t) -> (Int, Item t) -> IO ()
  mutTensor t (idx, value) = withForeignPtr t $ \tp -> Gen.c_set1d tp (fromIntegral idx) (translate value)
{-# NOINLINE genericFromList1d #-}

-- |Copy contents of tensor into a new one of specified size
genericResize
  :: forall t d . (THTensorMath' t, Num (HaskReal' t), k ~ Nat)
  => t -> Dim (d::[k]) -> t
genericResize t d = unsafePerformIO $ do
  let resDummy = genericNew d :: t
  newPtr <- withForeignPtr (getForeign t) Gen.c_newClone
  newFPtr <- newForeignPtr Gen.p_free newPtr
  with2ForeignPtrs newFPtr (getForeign resDummy) Gen.c_resizeAs
  pure $ construct newFPtr
{-# NOINLINE genericResize #-}

genericGet :: (THTensor' t, k ~ Nat) => (HaskReal (THForeignType t) -> Item t) -> Dim (d::[k]) -> t -> Item t
genericGet translate loc tensor = unsafePerformIO $ withForeignPtr
  (getForeign tensor)
  (\t -> pure . translate $ t `Gen.genericGet` loc)
{-# NOINLINE genericGet #-}

genericNewWithTensor :: (THTensor' t) => t -> t
genericNewWithTensor t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) Gen.c_newWithTensor
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericNewWithTensor #-}

-- |Create a new (double) tensor of specified dimensions and fill it with 0
genericNew
  :: (THTensorMath' t, Num (HaskReal' t), k ~ Nat)
  => Dim (d::[k]) -> t
genericNew dims = unsafePerformIO $ do
  newPtr <- Gen.constant dims 0
  fPtr <- newForeignPtr Gen.p_free newPtr
  void $ withForeignPtr fPtr Gen.fillZeros
  pure $ construct fPtr
{-# NOINLINE genericNew #-}

genericNew'
  :: (THTensorMath' t, Num (HaskReal' t), k ~ Nat)
  => SomeDims -> t
genericNew' (SomeDims ds) = genericNew ds

genericFree_ :: THForeignRef t => t -> IO ()
genericFree_ t = finalizeForeignPtr $! getForeign t

genericInit :: (THTensorMath' t, k ~ Nat) => (Item t -> HaskReal' t) -> Dim (d::[k]) -> Item t -> t
genericInit translate ds val = unsafePerformIO $ do
  newPtr <- Gen.constant ds (translate val)
  fPtr <- newForeignPtr Gen.p_free newPtr
  withForeignPtr fPtr (Gen.inplaceFill translate val)
  pure $ construct fPtr
{-# NOINLINE genericInit #-}

genericTranspose :: THTensor' t => Word -> Word -> t -> t
genericTranspose (fromIntegral->dim1C) (fromIntegral->dim2C) t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p dim1C dim2C)
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericTranspose #-}

genericTrans :: THTensor' t => t -> t
genericTrans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) (\p -> Gen.c_newTranspose p 1 0)
  newFPtr <- newForeignPtr Gen.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericTrans #-}

genericDynamicDims :: THTensor' t => t -> SomeDims
genericDynamicDims t = unsafePerformIO $
  withForeignPtr (getForeign t) (pure . Gen.getDynamicDim)
{-# NOINLINE genericDynamicDims #-}


with2ForeignPtrs :: ForeignPtr f0 -> ForeignPtr f1 -> (Ptr f0 -> Ptr f1 -> IO x) -> IO x
with2ForeignPtrs fp1 fp2 fn = withForeignPtr fp1 (withForeignPtr fp2 . fn)

with3ForeignPtrs :: ForeignPtr f0 -> ForeignPtr f1 -> ForeignPtr f2 -> (Ptr f0 -> Ptr f1 -> Ptr f2 -> IO x) -> IO x
with3ForeignPtrs fp0 fp1 fp2 fn =
  withForeignPtr fp0 (\p0 -> withForeignPtr fp1 (\p1 -> withForeignPtr fp2 (\p2 -> fn p0 p1 p2)))

with4ForeignPtrs :: ForeignPtr f0 -> ForeignPtr f1 -> ForeignPtr f2 -> ForeignPtr f3 -> (Ptr f0 -> Ptr f1 -> Ptr f2 -> Ptr f3 -> IO x) -> IO x
with4ForeignPtrs fp0 fp1 fp2 fp3 fn =
  withForeignPtr fp0 (\p0 -> withForeignPtr fp1 (\p1 -> withForeignPtr fp2 (\p2 -> withForeignPtr fp3 (\p3 -> fn p0 p1 p2 p3))))


with2THForeignRefs :: (THForeignRef f0, THForeignRef f1) => f0 -> f1 -> (Ptr (THForeignType f0) -> Ptr (THForeignType f1) -> IO x) -> IO x
with2THForeignRefs fp1 fp2 = with2ForeignPtrs (getForeign fp1) (getForeign fp2)

with3THForeignRefs :: (THForeignRef f0, THForeignRef f1, THForeignRef f2) => f0 -> f1 -> f2 -> (Ptr (THForeignType f0) -> Ptr (THForeignType f1) -> Ptr (THForeignType f2) -> IO x) -> IO x
with3THForeignRefs fp1 fp2 fp3 = with3ForeignPtrs (getForeign fp1) (getForeign fp2) (getForeign fp3)

with4THForeignRefs
  :: (THForeignRef f0, THForeignRef f1, THForeignRef f2, THForeignRef f3)
  => f0 -> f1 -> f2 -> f3
  -> (Ptr (THForeignType f0) -> Ptr (THForeignType f1) -> Ptr (THForeignType f2) -> Ptr (THForeignType f3) -> IO x)
  -> IO x
with4THForeignRefs a b c d = with4ForeignPtrs (getForeign a) (getForeign b) (getForeign c) (getForeign d)



