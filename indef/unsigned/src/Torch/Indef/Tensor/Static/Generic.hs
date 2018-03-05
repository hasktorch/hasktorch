{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Indef.Tensor.Static.Generic
  ( THSTensor'
  , THSTensorMath'
  , DownCastValueFn

  , THStaticRef(..)
  , TensorDoubleStatic(..)
  , TDS
  , TensorFloatStatic(..)
  , TFS

  , genericConstantIO
  , genericConstant
  , genericEqual
  , genericFromList1d
  , genericAsDynamic
  , genericFromDynamic
  , genericFromList
  , genericNewClone
  , genericDim
  , genericSomeDims
  , genericExpandMatrix
  ) where

import Control.Exception.Safe (MonadThrow, throwString)
import Control.Monad.Managed
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Singletons.Prelude.Num
import Foreign (Ptr)
import Foreign.C.Types (CLong)
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr, newForeignPtr)
import GHC.Exts
import System.IO.Unsafe (unsafePerformIO)

import Torch.Indef.StorageLong (newStorageLong)
import Torch.Indef.StorageTypes (StorageLong, StorageSize(..))
import Torch.Indef.Tensor.Types (THForeignRef, THForeignType, TensorFloat, TensorDouble(..))
import Torch.Dimensions (KnownDim, Dimensions, Dim, SomeDims(..), SingDimensions, KnownNatDim, dim)
import Torch.Raw.Tensor.Generic (Storage, CTorch.FFI.TH.Double.Tensor, CTorch.FFI.TH.Long.Storage, CTorch.FFI.TH.Float.Tensor, HaskReal)
import qualified Torch.Dimensions as Dim
import qualified Torch.Indef.Tensor.Types as Dyn (THForeignRef(..))
import qualified Torch.Indef.Tensor.Dynamic.Generic as Gen
import qualified Torch.Raw.Tensor.Generic as GenRaw

type THSTensor' t d = (GenRaw.THTensor (THForeignType (t d)), Num (HaskReal (t d)), SingDimensions d, THStaticRef t d)
type THSTensorMath' t d = (GenRaw.THTensorMath (THForeignType (t d)), THStaticRef t (d::[Nat]))
type DownCastValueFn t = (HaskReal t -> HaskReal (THForeignType t))

class THStaticRef t (d::[Nat]) where
  type family AsDynamic t d
  construct :: ForeignPtr (THForeignType (t d)) -> t d
  getForeign :: t d -> ForeignPtr (THForeignType (t d))
  asDynamic :: t d -> AsDynamic t d

-- ========================================================================= --
-- Types for static tensors

newtype TensorDoubleStatic (ds :: [Nat])
  = TDS { tdsTensor :: ForeignPtr CTorch.FFI.TH.Double.Tensor }
  deriving (Show)

type TDS = TensorDoubleStatic
type instance THForeignType (TensorDoubleStatic dim) = CTorch.FFI.TH.Double.Tensor
type instance Storage (TensorDoubleStatic (dim :: [Nat])) = StorageLong
type instance HaskReal (TensorDoubleStatic (dim :: [Nat])) = Double

instance Eq (TensorDoubleStatic d) where
  (==) = genericEqual

instance THStaticRef TensorDoubleStatic (dim :: [Nat]) where
  type AsDynamic TensorDoubleStatic dim = TensorDouble
  construct = TDS
  getForeign = tdsTensor
  asDynamic = genericAsDynamic

instance THForeignRef (TensorDoubleStatic (dim :: [Nat])) where
  construct = TDS
  getForeign = tdsTensor

-- Not sure how important this is
instance KnownNat l => IsList (TDS '[l]) where
  type Item (TDS '[l]) = Double
  fromList :: [Double] -> TDS '[l]
  fromList = genericFromList1d realToFrac
  toList t = unsafePerformIO (withForeignPtr (getForeign t) (pure . map realToFrac . GenRaw.flatten))
  {-# NOINLINE toList #-}


-------------------------------------------------------------------------------

newtype TensorFloatStatic (ds :: [Nat])
  = TFS { tfsTensor :: ForeignPtr CTorch.FFI.TH.Float.Tensor }
  deriving (Show)

type TFS = TensorFloatStatic
type instance THForeignType (TensorFloatStatic (dim :: [Nat])) = CTorch.FFI.TH.Float.Tensor
type instance Storage (TensorFloatStatic (dim :: [Nat])) = StorageLong
type instance HaskReal (TensorFloatStatic (dim :: [Nat])) = Integer

instance Eq (TensorFloatStatic d) where
  (==) = genericEqual

instance THStaticRef TensorFloatStatic (dim :: [Nat]) where
  type AsDynamic TensorFloatStatic dim = TensorFloat
  construct = TFS
  getForeign = tfsTensor
  asDynamic = genericAsDynamic

instance THForeignRef (TensorFloatStatic (dim :: [Nat])) where
  construct = TFS
  getForeign = tfsTensor

-- ========================================================================= --
-- Typeclasses for static tensors

class StaticTH t (d::[Nat]) where
  new :: Dim d -> t d
  new_ :: Dim d -> IO (t d)
  fromListNd :: Dim d -> [HaskReal (t d)] -> (t d)
  fromList1d :: [HaskReal (t d)] -> (t d)

  -- | create tensor of the same dimensions
  cloneDim :: t d -> t d -- takes unused argument, gets dimensions by matching types
  -- | create and initialize tensor
  init_ :: Double -> IO (t d)
  init :: Double -> t d
  -- | Display tensor
  printTensor :: t d -> IO ()

{-
  resize :: k ~ Nat => t -> Dim (d::[k]) -> t
  get :: k ~ Nat => Dim (d::[k]) -> t -> Item t
  newWithTensor :: t -> t
  free_ :: t -> IO ()
  init :: k ~ Nat => Dim (d::[k]) -> Item t -> t
  transpose :: Word -> Word -> t -> t
  trans :: t -> t
-}

-- not exported, these should be cleaned up since there is duplicate foreign pointer code in Tensor.Types
onForeign2Ptrs
  :: (THForeignRef a, THForeignRef b)
  => (Ptr (THForeignType a) -> Ptr (THForeignType b) -> x) -> a -> b -> x
onForeign2Ptrs fn = onForeign2PtrsIO (\pa pb -> pure $ fn pa pb)
{-# NOINLINE onForeign2Ptrs #-}

-- not exported, these should be cleaned up since there is duplicate foreign pointer code in Tensor.Types
onForeign2PtrsIO
  :: (THForeignRef a, THForeignRef b)
  => (Ptr (THForeignType a) -> Ptr (THForeignType b) -> IO x) -> a -> b -> x
onForeign2PtrsIO fn a b = unsafePerformIO $
  Gen.with2ForeignPtrs (Dyn.getForeign a) (Dyn.getForeign b) fn
{-# NOINLINE onForeign2PtrsIO #-}


-- ========================================================================= --
-- the more important generic functions

-- | unsafePerformIO call to 'genericConstantIO'
genericConstant
  :: (THSTensor' t d, THSTensorMath' t d)
  => DownCastValueFn (t d) -> Dim d -> HaskReal (t d) -> t d
genericConstant a b c = unsafePerformIO (genericConstantIO a b c)
{-# NOINLINE genericConstant #-}

-- | Make a foreign pointer from requested dimensions
genericConstantIO
  :: (THSTensor' t d, THSTensorMath' t d)
  => DownCastValueFn (t d) -> Dim d -> HaskReal (t d) -> IO (t d)
genericConstantIO cast dim value = do
  newPtr <- GenRaw.constant dim (cast value)
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ construct fPtr


-- ========================================================================= --
-- Generic helper functions

-- | Generic equality
genericEqual :: (THForeignRef t, Gen.THTensorMath' t) => t -> t -> Bool
genericEqual = onForeign2Ptrs GenRaw.genericEqual


-- TODO: try to force strict evaluation to avoid potential FFI + IO + mutation bugs.
-- however `go` never executes with deepseq: else unsafePerformIO $ pure (deepseq go result)
genericFromList1d
  :: forall t n . (KnownNatDim n, THSTensor' t '[n], THSTensorMath' t '[n])
  => (DownCastValueFn (t '[n])) -> [HaskReal (t '[n])] -> t '[n]
genericFromList1d cast l =
  if fromIntegral (natVal (Proxy :: Proxy n)) /= length l
  then error "List length does not match tensor dimensions"
  else unsafePerformIO $ do
    res <- genericConstantIO cast (dim :: Dim '[n]) (0 :: HaskReal (t '[n]))
    mapM_ (upd res) (zip [0..length l - 1] l)
    pure res
 where
  upd :: t '[n] -> (Int, HaskReal (t '[n])) -> IO ()
  upd t (idx, value) = withForeignPtr (getForeign t) $ \tp ->
    GenRaw.c_set1d tp (fromIntegral idx) (cast value)
{-# NOINLINE genericFromList1d #-}

genericAsDynamic
  :: (THForeignRef dyn, THStaticRef s dim, THForeignType (s dim) ~ THForeignType dyn)
  => s dim -> dyn
genericAsDynamic = Dyn.construct . getForeign

-- | TODO: add dimension check
genericFromDynamic
  :: (THForeignRef dyn, THStaticRef s dim, THForeignType (s dim) ~ THForeignType dyn, Gen.THTensor' dyn)
  => dyn -> s dim
genericFromDynamic dyn = unsafePerformIO $ do
  newPtr <- withForeignPtr (Dyn.getForeign dyn) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericFromDynamic #-}

-- |Make a resized tensor
genericResize
  :: forall t d1 d2 . (THStaticRef t d1, THSTensorMath' t d2, THSTensor' t d2)
  => (THForeignType (t d1) ~ THForeignType (t d2))
  => DownCastValueFn (t d2) -> t d1 -> t d2
genericResize cast t = unsafePerformIO $ do
  resDummy :: t d2 <- genericConstantIO cast (dim :: Dim d2) (0 :: HaskReal (t d2))
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  Gen.with2ForeignPtrs newFPtr (getForeign resDummy) GenRaw.c_resizeAs
  pure (construct newFPtr)
{-# NOINLINE genericResize #-}

-- | Initialize a tensor of arbitrary dimension from a list
genericFromList
  :: forall t d
  -- both desired dimensions, and 1-d version are static tensors
  . (THSTensorMath' t d, THSTensor' t d, THSTensorMath' t '[Product d], THSTensor' t '[Product d])
  -- both desired tensor, and 1-d version have the same internal type families
  => (THForeignType (t d) ~ THForeignType (t '[Product d]), HaskReal (t d) ~ HaskReal (t '[Product d]))
  -- our 1-d cast (which gets resized) is known at compile time
  => (KnownNatDim (Product d))
  -- finally, take a casting function, a list of haskell-typed nums, and build any tensor
  => DownCastValueFn (t d) -> [HaskReal (t d)] -> t d
genericFromList cast l = genericResize cast (genericFromList1d cast l :: t '[Product d])

genericNewClone :: THSTensor' t d => t d -> t d
genericNewClone t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericNewClone #-}

-- -- |generalized transpose - needs type level determination of perturbed dimensions
-- generictranspose :: Word -> Word -> TensorDoubleStatic d1 -> TensorDoubleStatic d2
-- generictranspose = undefined

-- | matrix specialization of transpose transpose
genericTrans
  :: forall t r c . (THSTensor' t '[r, c], THSTensor' t '[c, r])
  => (THForeignType (t '[r, c]) ~ THForeignType (t '[c, r]))
  => t '[r, c] -> t '[c, r]
genericTrans t = unsafePerformIO $ do
  newPtr  <- withForeignPtr (getForeign t) (\tPtr -> GenRaw.c_newTranspose tPtr 1 0)
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ construct newFPtr
{-# NOINLINE genericTrans #-}

genericDim :: Dimensions d => TensorDoubleStatic d -> Dim d
genericDim _ = Dim.dim

genericSomeDims :: Dimensions d => TensorDoubleStatic (d::[Nat]) -> SomeDims
genericSomeDims = SomeDims . genericDim


-- | Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
genericExpandMatrix
  :: forall t n1 n2 . (KnownNatDim n1, KnownNatDim n2)
  => (THSTensor' t '[n2, n1], THSTensor' t '[n1])
  => (THSTensorMath' t '[n1])
  => (THForeignType (t '[n2, n1]) ~ THForeignType (t '[n1]))
  => DownCastValueFn (t '[n2, n1]) -> t '[n1] -> t '[n2, n1]
genericExpandMatrix cast t = unsafePerformIO $ do
  r_ <- genericConstantIO cast (dim :: Dim '[n2, n1]) 0
  _withManaged3 GenRaw.c_expand r_ t s
  pure r_
  where
    s1, s2 :: Int
    s1 = fromIntegral $ natVal (Proxy :: Proxy n1)
    s2 = fromIntegral $ natVal (Proxy :: Proxy n2)

    s :: StorageLong
    s = newStorageLong (S2 (s2, s1))
{-# NOINLINE genericExpandMatrix #-}

-- not exported, these should be cleaned up since there is duplicate foreign pointer code in Tensor.Types
_withManaged3
  :: (THSTensor' t a, THSTensor' t b)
  => (Ptr (THForeignType (t a)) -> Ptr (THForeignType (t b)) -> Ptr CTorch.FFI.TH.Long.Storage -> IO ())
  -> t a -> t b -> StorageLong -> IO ()
_withManaged3 fn a b c = runManaged $ do
  a' <- managed (withForeignPtr (getForeign a))
  b' <- managed (withForeignPtr (getForeign b))
  c' <- managed (withForeignPtr (Dyn.getForeign c))
  liftIO (fn a' b' c')


