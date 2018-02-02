{-# OPTIONS_GHC -fno-cse -fno-full-laziness #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
module Torch.Core.Tensor.Static.Long
  ( StaticTensor(..)
  , tls_dim
  , tls_expand
  , tls_new -- *
  , tls_new_ -- *
  , tls_fromDynamic
  , tls_fromList
  , tls_init -- *
  , tls_cloneDim -- *
  , tls_newClone
  , tls_p -- *
  , tls_resize
  , tls_toDynamic
  , tls_trans -- matrix specialization of transpose
  , TensorLongStatic(..)
  , TLS(..)
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
import Torch.Core.Tensor.Types (THForeignRef(..), THForeignType, TensorLong(..), TensorLong(..))
import Torch.Core.Tensor.Dim (KnownDim, Dimensions, Dim, SomeDims(..))
import Torch.Raw.Tensor.Generic (Storage, CTHLongTensor, CTHDoubleTensor, CTHLongStorage)
import qualified Torch.Core.Tensor.Dim as Dim
import qualified Torch.Core.Tensor.Dynamic.Generic as Gen
import qualified Torch.Raw.Tensor.Generic as GenRaw

import THLongTensor (c_THLongTensor_set1d)

-- ========================================================================= --
-- Types for static tensors

newtype TensorLongStatic (ds :: [Nat])
  = TLS { tlsTensor :: ForeignPtr CTHLongTensor }
  deriving (Show)

type TLS = TensorLongStatic
type instance THForeignType (TensorLongStatic (dim :: [Nat])) = CTHLongTensor
type instance Storage (TensorLongStatic (dim :: [Nat])) = StorageLong

instance Eq (TensorLongStatic d) where
  (==) t1 t2 = unsafePerformIO $
    Gen.with2ForeignPtrs
      (tlsTensor t1)
      (tlsTensor t2)
      (\t1c t2c -> pure . (== 1) $ GenRaw.c_equal t1c t2c)
  {-# NOINLINE (==) #-}

instance THForeignRef (TensorLongStatic (dim :: [Nat])) where
  construct = TLS
  getForeign = tlsTensor

-- TODO: try to force strict evaluation to avoid potential FFI + IO + mutation bugs.
-- however `go` never executes with deepseq: else unsafePerformIO $ pure (deepseq go result)
fromList1d :: forall n . (KnownNat n, KnownDim n) => [Integer] -> TensorLongStatic '[n]
fromList1d l =
  if fromIntegral (natVal (Proxy :: Proxy n)) /= length l
  then error "List length does not match tensor dimensions"
  else unsafePerformIO $ do
    mapM_ (upd res) (zip [0..length l - 1] l)
    pure res
 where
  res :: TLS '[n]
  res = tls_new

  upd :: TLS '[n] -> (Int, Integer) -> IO ()
  upd t (idx, value) = withForeignPtr (tlsTensor t) (\tp -> GenRaw.c_set1d tp (fromIntegral idx) ( fromInteger value ) )
{-# NOINLINE fromList1d #-}

-- ========================================================================= --


-- TODO: get rid of this double-specific typeclass and just extend functionality
-- as independent functions using singletons
class StaticTensor t where
  -- | tensor dimensions
  -- | create tensor
  tls_new_ :: IO t
  tls_new :: t
  -- | create tensor of the same dimensions
  tls_cloneDim :: t -> t -- takes unused argument, gets dimensions by matching types
  -- | create and initialize tensor
  tls_init_ :: Integer -> IO t
  tls_init :: Integer -> t
  -- | Display tensor
  tls_p ::  t -> IO ()

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
instance KnownNat l => IsList (TLS '[l]) where
  type Item (TLS '[l]) = Integer
  fromList l = if (fromIntegral $ natVal (Proxy :: Proxy l)) /= length l
               then error "List length does not match tensor dimensions"
               else unsafePerformIO $ go result
               -- TODO: try to force strict evaluation
               -- to avoid potential FFI + IO + mutation bugs.
               -- however `go` never executes with deepseq:
               -- else unsafePerformIO $ pure (deepseq go result)
    where
      result = tls_new
      go t = do
        mapM_ mutTensor (zip [0..(length l) - 1] l)
        pure t
        where
          mutTensor (idx, value) =
            let (idxC, valueC) = (fromIntegral idx, fromInteger value) in
              withForeignPtr (tlsTensor t)
                (\tp -> do
                    -- print idx -- check to see when mutation happens
                    c_THLongTensor_set1d tp idxC valueC
                )
  -- toList t = unsafePerformIO (withForeignPtr (getForeign t) (pure . map fromInteger . GenRaw.flatten) )
  -- {-# NOINLINE toList #-}

-- | Initialize a 1D tensor from a list
tls_fromList1D :: KnownNat n => [Integer] -> TLS '[n]
tls_fromList1D l = fromList l

-- |Initialize a tensor of arbitrary dimension from a list
tls_fromList
  :: forall d2 . (Dimensions d2, SingI '[Product d2], SingI d2,
                  KnownNat (Product d2), KnownDim (Product d2))
  => [Integer]
  -> TLS d2
tls_fromList l = tls_resize (tls_fromList1D l :: TLS '[Product d2])

-- |Make a resized tensor
tls_resize
  :: forall d1 d2. (Product d1 ~ Product d2, Dimensions d1, Dimensions d2,
                    SingI d1, SingI d2)
  => TLS d1 -> TLS d2
tls_resize t = unsafePerformIO $ do
  let resDummy = tls_new :: TLS d2
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  Gen.with2ForeignPtrs newFPtr (getForeign resDummy) GenRaw.c_resizeAs
  pure (TLS newFPtr)
{-# NOINLINE tls_resize #-}

tls_toDynamic :: TensorLongStatic d -> TensorLong
tls_toDynamic (TLS fp) = TensorLong fp


-- |TODO: add dimension check
tls_fromDynamic :: SingI d => TensorLong -> TensorLongStatic d
tls_fromDynamic t = unsafePerformIO $ do
  newPtr :: Ptr CTHLongTensor <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr :: ForeignPtr CTHLongTensor <- newForeignPtr GenRaw.p_free newPtr
  pure $ TLS newFPtr
{-# NOINLINE tls_fromDynamic #-}

tls_newClone :: TensorLongStatic d -> TensorLongStatic d
tls_newClone t = unsafePerformIO $ do
  newPtr <- withForeignPtr (getForeign t) GenRaw.c_newClone
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ TLS newFPtr
{-# NOINLINE tls_newClone #-}

-- -- |generalized transpose - needs type level determination of perturbed dimensions
-- tls_transpose :: Word -> Word -> TensorLongStatic d1 -> TensorLongStatic d2
-- tls_transpose = undefined

-- |matrix specialization of transpose transpose
tls_trans :: TensorLongStatic '[r, c] -> TensorLongStatic '[c, r]
tls_trans t = unsafePerformIO $ do
  newPtr <- withForeignPtr (tlsTensor t) (\tPtr -> GenRaw.c_newTranspose tPtr 1 0)
  newFPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ TLS newFPtr
{-# NOINLINE tls_trans #-}

-- | FIXME: this functionality is broken
tls_dim :: Dimensions d => TensorLongStatic d -> Dim d
tls_dim _ = Dim.dim

tls_someDims :: Dimensions d => TensorLongStatic (d::[Nat]) -> SomeDims
tls_someDims = SomeDims . tls_dim

-- |Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
tls_expand :: forall d1 d2 . (KnownDim d1, KnownDim d2, KnownNat d1, KnownNat d2) => TLS '[d1] -> TLS '[d2, d1]
tls_expand t = unsafePerformIO $ do
  let r_ = tls_init 0 --  :: TLS '[d2, d1]
  _withManaged3 GenRaw.c_expand r_ t s -- (tlsTensor r_) (tlsTensor t) (slStorage s)
  pure r_
  where
    s1, s2 :: Int
    s1 = fromIntegral $ natVal (Proxy :: Proxy d1)
    s2 = fromIntegral $ natVal (Proxy :: Proxy d2)

    s :: StorageLong
    s = newStorageLong (S2 (s2, s1))
{-# NOINLINE tls_expand #-}

_withManaged3
  :: (Ptr CTHLongTensor -> Ptr CTHLongTensor -> Ptr CTHLongStorage -> IO ())
  -> TLS (a::[Nat]) -> TLS (b::[Nat]) -> StorageLong -> IO ()
_withManaged3 fn a b c = runManaged $ do
  a' <- managed (withForeignPtr (getForeign a))
  b' <- managed (withForeignPtr (getForeign b))
  c' <- managed (withForeignPtr (getForeign c))
  liftIO (fn a' b' c')

-- | unsafePerformIO call to 'mkTHelperIO'
mkTHelper :: Dim (d::[Nat]) -> (ForeignPtr CTHLongTensor -> TLS d) -> Integer -> TLS d
mkTHelper a b c = unsafePerformIO (mkTHelperIO a b c)
{-# NOINLINE mkTHelper #-}

-- | Make a foreign pointer from requested dimensions
mkTHelperIO :: Dim (d::[Nat]) -> (ForeignPtr CTHLongTensor -> TLS d) -> Integer -> IO (TLS d)
mkTHelperIO dim makeStatic value = do
  newPtr <- GenRaw.constant dim (fromInteger value)
  fPtr <- newForeignPtr GenRaw.p_free newPtr
  pure $ makeStatic fPtr

instance (Dimensions d, SingI d) => StaticTensor (TensorLongStatic d)  where
  tls_init = mkTHelper (Dim.dim :: Dim d) (\fp -> TLS fp :: TLS d)
  tls_init_ = mkTHelperIO (Dim.dim :: Dim d) (\fp -> TLS fp :: TLS d)
  tls_new = tls_init 0
  tls_new_ = tls_init_ 0
  tls_cloneDim _ = tls_new :: TLS d
  tls_p tensor = withForeignPtr (tlsTensor tensor) GenRaw.dispRaw
