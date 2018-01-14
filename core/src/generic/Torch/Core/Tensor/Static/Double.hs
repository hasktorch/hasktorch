{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
module Torch.Core.Tensor.Static.Double
  ( StaticTensor(..)
  , tds_dim
  , tds_expand
  , tds_new -- *
  , tds_new_ -- *
  , tds_fromDynamic
  , tds_fromList
  , tds_init -- *
  , tds_cloneDim -- *
  , tds_newClone
  , tds_p -- *
  , tds_resize
  , tds_toDynamic
  , tds_trans -- matrix specialization of transpose
  , TensorDoubleStatic(..)
  , TDS(..)
  , Nat
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

import Torch.Core.StorageLong (newStorageLong)
import Torch.Core.StorageTypes (StorageLong, StorageSize(..))
import Torch.Core.Tensor.Types (THForeignRef(..), THForeignType, TensorDouble(..))
import Torch.Core.Tensor.Dim (KnownDim, Dimensions, Dim, SomeDims(..))
import Torch.Raw.Tensor.Generic (Storage, CTHDoubleTensor, CTHLongStorage)
import qualified Torch.Core.Tensor.Dim as Dim
import qualified Torch.Core.Tensor.Dynamic.Generic as Gen
import qualified Torch.Raw.Tensor.Generic as GenRaw

import THDoubleTensor (c_THDoubleTensor_set1d)

-- ========================================================================= --
-- Types for static tensors

newtype TensorDoubleStatic (ds :: [Nat])
  = TDS { tdsTensor :: ForeignPtr CTHDoubleTensor }
  deriving (Show)

type TDS = TensorDoubleStatic
type instance THForeignType (TensorDoubleStatic (dim :: [Nat])) = CTHDoubleTensor
type instance Storage (TensorDoubleStatic (dim :: [Nat])) = StorageLong

instance Eq (TensorDoubleStatic d) where
  (==) t1 t2 = unsafePerformIO $
    Gen.with2ForeignPtrs
      (tdsTensor t1)
      (tdsTensor t2)
      (\t1c t2c -> pure . (== 1) $ GenRaw.c_equal t1c t2c)
  {-# NOINLINE (==) #-}

instance THForeignRef (TensorDoubleStatic (dim :: [Nat])) where
  construct = TDS
  getForeign = tdsTensor

-- TODO: try to force strict evaluation to avoid potential FFI + IO + mutation bugs.
-- however `go` never executes with deepseq: else unsafePerformIO $ pure (deepseq go result)
fromList1d :: forall n . (KnownNat n, KnownDim n) => [Double] -> TensorDoubleStatic '[n]
fromList1d l =
  if fromIntegral (natVal (Proxy :: Proxy n)) /= length l
  then error "List length does not match tensor dimensions"
  else unsafePerformIO $ do
    mapM_ (upd res) (zip [0..length l - 1] l)
    pure res
 where
  res :: TDS '[n]
  res = tds_new

  upd :: TDS '[n] -> (Int, Double) -> IO ()
  upd t (idx, value) = withForeignPtr (tdsTensor t) (\tp -> GenRaw.c_set1d tp (fromIntegral idx) (realToFrac value))
{-# NOINLINE fromList1d #-}

-- ========================================================================= --


-- TODO: get rid of this double-specific typeclass and just extend functionality
-- as independent functions using singletons
class StaticTensor t where
  -- | tensor dimensions
  -- | create tensor
  tds_new_ :: IO t
  tds_new :: t
  -- | create tensor of the same dimensions
  tds_cloneDim :: t -> t -- takes unused argument, gets dimensions by matching types
  -- | create and initialize tensor
  tds_init_ :: Double -> IO t
  tds_init :: Double -> t
  -- | Display tensor
  tds_p ::  t -> IO ()

{-
class IsList t => StaticTH t where
  printTensor :: t -> IO ()
  fromListNd :: k ~ Nat => Dim (d::[k]) -> [Item t] -> t
  fromList1d :: [Item t] -> t
  resize :: k ~ Nat => t -> Dim (d::[k]) -> t
  get :: k ~ Nat => Dim (d::[k]) -> t -> Item t
  newWithTensor :: t -> t
  new :: k ~ Nat => Dim (d::[k]) -> t
  new_ :: k ~ Nat => Dim (d::[k]) -> IO t
  free_ :: t -> IO ()
  init :: k ~ Nat => Dim (d::[k]) -> Item t -> t
  transpose :: Word -> Word -> t -> t
  trans :: t -> t
-}


-- Not sure how important this is
instance KnownNat l => IsList (TDS '[l]) where
  type Item (TDS '[l]) = Double
  fromList l = if (fromIntegral $ natVal (Proxy :: Proxy l)) /= length l
               then error "List length does not match tensor dimensions"
               else unsafePerformIO $ go result
               -- TODO: try to force strict evaluation
               -- to avoid potential FFI + IO + mutation bugs.
               -- however `go` never executes with deepseq:
               -- else unsafePerformIO $ pure (deepseq go result)
    where
      result = tds_new
      go t = do
        mapM_ mutTensor (zip [0..(length l) - 1] l)
        pure t
        where
          mutTensor (idx, value) =
            let (idxC, valueC) = (fromIntegral idx, realToFrac value) in
              withForeignPtr (tdsTensor t)
                (\tp -> do
                    -- print idx -- check to see when mutation happens
                    c_THDoubleTensor_set1d tp idxC valueC
                )
  toList t = unsafePerformIO (withForeignPtr (getForeign t) (pure . map realToFrac . GenRaw.flatten))
  {-# NOINLINE toList #-}

-- | Initialize a 1D tensor from a list
tds_fromList1D :: KnownNat n => [Double] -> TDS '[n]
tds_fromList1D l = fromList l

-- |Initialize a tensor of arbitrary dimension from a list
tds_fromList
  :: forall d2 . (Dimensions d2, SingI '[Product d2], SingI d2,
                  KnownNat (Product d2), KnownDim (Product d2))
  => [Double]
  -> TDS d2
tds_fromList l = tds_resize (tds_fromList1D l :: TDS '[Product d2])

-- |Make a resized tensor
tds_resize
  :: forall d1 d2. (Product d1 ~ Product d2, Dimensions d1, Dimensions d2,
                    SingI d1, SingI d2)
  => TDS d1 -> TDS d2
tds_resize t = unsafePerformIO $ do
  let resDummy = tds_new :: TDS d2
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  Gen.with2ForeignPtrs newFPtr (getForeign resDummy) GenRaw.c_resizeAs
  pure (TDS newFPtr)
{-# NOINLINE tds_resize #-}

tds_toDynamic :: TensorDoubleStatic d -> TensorDouble
tds_toDynamic (TDS fp) = TensorDouble fp


-- |TODO: add dimension check
tds_fromDynamic :: SingI d => TensorDouble -> TensorDoubleStatic d
tds_fromDynamic t = unsafePerformIO $ do
  newPtr :: Ptr CTHDoubleTensor <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr :: ForeignPtr CTHDoubleTensor <- newForeignPtr GenRaw.p_free newPtr
  pure $ TDS newFPtr
{-# NOINLINE tds_fromDynamic #-}

tds_newClone :: TensorDoubleStatic d -> TensorDoubleStatic d
tds_newClone t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ TDS newFPtr
{-# NOINLINE tds_newClone #-}

-- -- |generalized transpose - needs type level determination of perturbed dimensions
-- tds_transpose :: Word -> Word -> TensorDoubleStatic d1 -> TensorDoubleStatic d2
-- tds_transpose = undefined

-- |matrix specialization of transpose transpose
tds_trans :: TensorDoubleStatic '[r, c] -> TensorDoubleStatic '[c, r]
tds_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tdsTensor t) (\tPtr -> GenRaw.c_newTranspose tPtr 1 0)
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ TDS newFPtr
{-# NOINLINE tds_trans #-}

-- | FIXME: this functionality is broken
tds_dim :: Dimensions d => TensorDoubleStatic d -> Dim d
tds_dim _ = Dim.dim

tds_someDims :: Dimensions d => TensorDoubleStatic (d::[Nat]) -> SomeDims
tds_someDims = SomeDims . tds_dim

-- |Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
tds_expand :: forall d1 d2 . (KnownDim d1, KnownDim d2, KnownNat d1, KnownNat d2) => TDS '[d1] -> TDS '[d2, d1]
tds_expand t = unsafePerformIO $ do
  let r_ = tds_init 0 --  :: TDS '[d2, d1]
  _withManaged3 GenRaw.c_expand r_ t s -- (tdsTensor r_) (tdsTensor t) (slStorage s)
  pure r_
  where
    s1, s2 :: Int
    s1 = fromIntegral $ natVal (Proxy :: Proxy d1)
    s2 = fromIntegral $ natVal (Proxy :: Proxy d2)

    s :: StorageLong
    s = newStorageLong (S2 (s2, s1))
{-# NOINLINE tds_expand #-}

_withManaged3
  :: (Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> Ptr CTHLongStorage -> IO ())
  -> TDS (a::[Nat]) -> TDS (b::[Nat]) -> StorageLong -> IO ()
_withManaged3 fn a b c = runManaged $ do
  a' <- managed (withForeignPtr (getForeign a))
  b' <- managed (withForeignPtr (getForeign b))
  c' <- managed (withForeignPtr (getForeign c))
  liftIO (fn a' b' c')

-- | unsafePerformIO call to 'mkTHelperIO'
mkTHelper :: Dim (d::[Nat]) -> (ForeignPtr CTHDoubleTensor -> TDS d) -> Double -> TDS d
mkTHelper a b c = unsafePerformIO (mkTHelperIO a b c)
{-# NOINLINE mkTHelper #-}

-- | Make a foreign pointer from requested dimensions
mkTHelperIO :: Dim (d::[Nat]) -> (ForeignPtr CTHDoubleTensor -> TDS d) -> Double -> IO (TDS d)
mkTHelperIO dim makeStatic value = do
  newPtr <- GenRaw.constant dim (realToFrac value)
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ makeStatic fPtr

instance (Dimensions d, SingI d) => StaticTensor (TensorDoubleStatic d)  where
  tds_init = mkTHelper (Dim.dim :: Dim d) (\fp -> TDS fp :: TDS d)
  tds_init_ = mkTHelperIO (Dim.dim :: Dim d) (\fp -> TDS fp :: TDS d)
  tds_new = tds_init 0.0
  tds_new_ = tds_init_ 0.0
  tds_cloneDim _ = tds_new :: TDS d
  tds_p tensor = withForeignPtr (tdsTensor tensor) GenRaw.dispRaw

