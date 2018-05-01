{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.Tensor where

import Control.Exception.Safe
import Control.Monad.Trans
import Control.Monad.Trans.Maybe
import Data.Coerce
import Data.Maybe
import GHC.Natural
import System.IO.Unsafe

import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Index
import Torch.Indef.Static.Tensor.Copy
import qualified Torch.Indef.Dynamic.Tensor as Dynamic
import qualified Torch.Types.TH as TH
import qualified Torch.FFI.TH.Long.Storage as TH
import qualified Torch.Sig.Types as Sig

instance Show (Tensor (d::[Nat])) where
  show t = unsafePerformIO $ do
    SomeDims ds <- getDims t
    (vs, desc) <- Dynamic.showTensor (get1d t) (get2d t) (get3d t) (get4d t) (dimVals ds)
    pure (vs ++ "\n" ++ desc)
  {-# NOINLINE show #-}

-- FIXME: Definitely don't export this. Make sure these gory details never see the day of light.
sudo :: Tensor d -> Tensor d'
sudo t = Sig.asStatic ((Sig.asDynamic t) :: Dynamic)

-- | same as 'Dynamic.isSameSizeAs' but only uses type-level dimensions to compute.
isSameSizeAs :: forall d d' . (Dimensions d', Dimensions d) => Tensor d -> Tensor d' -> Bool
isSameSizeAs _ _ = dimVals (dim :: Dim d) == dimVals (dim :: Dim d')

vector :: forall n . KnownNat n => [HsReal] -> Maybe (Tensor '[n])
vector rs
  | genericLength rs == natVal (Proxy :: Proxy n) = Just . asStatic . Dynamic.vector $ rs
  | otherwise = Nothing

newExpand t = fmap asStatic . Dynamic.newExpand (asDynamic t)
_expand r t = Dynamic._expand (asDynamic r) (asDynamic t)
_expandNd rs os = Dynamic._expandNd (fmap asDynamic rs) (fmap asDynamic os)

_resize t a b = Dynamic._resize (asDynamic t) a b >> pure (sudo t)
_resize1d t a = Dynamic._resize1d (asDynamic t) a >> pure (sudo t)
_resize2d t a b = Dynamic._resize2d (asDynamic t) a b >> pure (sudo t)
_resize3d t a b c = Dynamic._resize3d (asDynamic t) a b c >> pure (sudo t)
_resize4d t a b c d = Dynamic._resize4d (asDynamic t) a b c d >> pure (sudo t)
_resize5d t a b c d e = Dynamic._resize5d (asDynamic t) a b c d e >> pure (sudo t)
_resizeAs src tar = Dynamic._resizeAs (asDynamic src) (asDynamic tar) >> pure (sudo src)
_resizeNd src a b c = Dynamic._resizeNd (asDynamic src) a b c >> pure (sudo src)
retain t = Dynamic.retain (asDynamic t)
_clearFlag t = Dynamic._clearFlag (asDynamic t)
tensordata t = Dynamic.tensordata (asDynamic t)
get1d t = Dynamic.get1d (asDynamic t)
get2d t = Dynamic.get2d (asDynamic t)
get3d t = Dynamic.get3d (asDynamic t)
get4d t = Dynamic.get4d (asDynamic t)
isContiguous t = Dynamic.isContiguous (asDynamic t)

isSetTo t0 t1 = Dynamic.isSetTo (asDynamic t0) (asDynamic t1)
isSize t = Dynamic.isSize (asDynamic t)
nDimension t = Dynamic.nDimension (asDynamic t)
nElement t = Dynamic.nElement (asDynamic t)
_narrow t0 t1 = Dynamic._narrow (asDynamic t0) (asDynamic t1)

-- | renamed from TH's @new@ because this always returns an empty tensor
-- FIXME: this __technically should be @IO (Tensor '[])@, but if you leave it as-is
-- the types line-up nicely (and we currently don't use rank-0 tensors).
empty = asStatic <$> Dynamic.empty

newClone t = asStatic <$> Dynamic.newClone (asDynamic t)
newContiguous t = asStatic <$> Dynamic.newContiguous (asDynamic t)
newNarrow t a b c = asStatic <$> Dynamic.newNarrow (asDynamic t) a b c
newSelect t a b = asStatic <$> Dynamic.newSelect (asDynamic t) a b
newSizeOf t = Dynamic.newSizeOf (asDynamic t)
newStrideOf t = Dynamic.newStrideOf (asDynamic t)
newTranspose t a b = asStatic <$> Dynamic.newTranspose (asDynamic t) a b
newUnfold t a b c = asStatic <$> Dynamic.newUnfold (asDynamic t) a b c
newView t a = asStatic <$> Dynamic.newView (asDynamic t) a
newWithSize a0 a1 = asStatic <$> Dynamic.newWithSize a0 a1
newWithSize1d a0 = asStatic <$> Dynamic.newWithSize1d a0
newWithSize2d a0 a1 = asStatic <$> Dynamic.newWithSize2d a0 a1
newWithSize3d a0 a1 a2 = asStatic <$> Dynamic.newWithSize3d a0 a1 a2
newWithSize4d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithSize4d a0 a1 a2 a3
newWithStorage a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage a0 a1 a2 a3
newWithStorage1d a0 a1 a2 = asStatic <$> Dynamic.newWithStorage1d a0 a1 a2
newWithStorage2d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage2d a0 a1 a2 a3
newWithStorage3d a0 a1 a2 a3 a4 = asStatic <$> Dynamic.newWithStorage3d a0 a1 a2 a3 a4
newWithStorage4d a0 a1 a2 a3 a4 a5 = asStatic <$> Dynamic.newWithStorage4d a0 a1 a2 a3 a4 a5
newWithTensor t = asStatic <$> Dynamic.newWithTensor (asDynamic t)
_select t0 t1 = Dynamic._select (asDynamic t0) (asDynamic t1)
_set t0 t1 = Dynamic._set (asDynamic t0) (asDynamic t1)
_set1d t = Dynamic._set1d (asDynamic t)
_set2d t = Dynamic._set2d (asDynamic t)
_set3d t = Dynamic._set3d (asDynamic t)
_set4d t = Dynamic._set4d (asDynamic t)
_setFlag t = Dynamic._setFlag (asDynamic t)
_setStorage t = Dynamic._setStorage (asDynamic t)
_setStorage1d t = Dynamic._setStorage1d (asDynamic t)
_setStorage2d t = Dynamic._setStorage2d (asDynamic t)
_setStorage3d t = Dynamic._setStorage3d (asDynamic t)
_setStorage4d t = Dynamic._setStorage4d (asDynamic t)
_setStorageNd t = Dynamic._setStorageNd (asDynamic t)
size t = Dynamic.size (asDynamic t)
sizeDesc t = Dynamic.sizeDesc (asDynamic t)
_squeeze t0 t1 = Dynamic._squeeze (asDynamic t0) (asDynamic t1)
_squeeze1d t0 t1 = Dynamic._squeeze1d (asDynamic t0) (asDynamic t1)
storage t = Dynamic.storage (asDynamic t)
storageOffset t = Dynamic.storageOffset (asDynamic t)
stride t = Dynamic.stride (asDynamic t)
_transpose t0 t1 = Dynamic._transpose (asDynamic t0) (asDynamic t1)
_unfold t0 t1 = Dynamic._unfold (asDynamic t0) (asDynamic t1)
_unsqueeze1d t0 t1 = Dynamic._unsqueeze1d (asDynamic t0) (asDynamic t1)

-- ========================================================================= --

shape :: Tensor d -> IO [Size]
shape t = do
  ds <- nDimension t
  mapM (size t . fromIntegral) [0..ds-1]

withNew :: forall d . (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
withNew op = new >>= \r -> op r >> pure r

-- Should be renamed to @newFromSize@
withEmpty :: forall d . (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
withEmpty op = new >>= \r -> op r >> pure r

-- We can get away with this some of the time, when Torch does the resizing in C, but you need to look at
-- the c implementation
withEmpty' :: (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
withEmpty' op = empty >>= \r -> op r >> pure r

-- This is actually 'inplace'. Dimensions may change from original tensor given Torch resizing.
withInplace :: (Dimensions d) => Tensor d -> (Tensor d -> Tensor d -> IO ()) -> IO (Tensor d)
withInplace t op = op t t >> pure t

-- This is actually 'inplace'. Dimensions may change from original tensor given Torch resizing.
sudoInplace
  :: forall d d'
  .  Tensor d -> (Tensor d' -> Tensor d -> IO ()) -> IO (Tensor d')
sudoInplace t op = op ret t >> pure ret
  where
    ret :: Tensor d'
    ret = asStatic . asDynamic $ t

throwFIXME :: MonadThrow io => String -> String -> io x
throwFIXME fixme msg = throwString $ msg ++ " (FIXME: " ++ fixme ++ ")"

throwNE :: MonadThrow io => String -> io x
throwNE = throwFIXME "make this function only take a non-empty [Nat]"

throwGT4 :: MonadThrow io => String -> io x
throwGT4 fnname = throwFIXME
  ("review how TH supports `" ++ fnname ++ "` operations on > rank-4 tensors")
  (fnname ++ " with >4 rank")


_setStorageDim :: Tensor d -> Storage -> StorageOffset -> [(Size, Stride)] -> IO ()
_setStorageDim t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> _setStorage1d t s o x
  [x, y]       -> _setStorage2d t s o x y
  [x, y, z]    -> _setStorage3d t s o x y z
  [x, y, z, q] -> _setStorage4d t s o x y z q
  _            -> throwGT4 "setStorage"

_setDim :: forall d d' . (Dimensions d') => Tensor d -> Dim d' -> HsReal -> IO ()
_setDim t d v = case dimVals d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> _set1d t x       v
  [x, y]       -> _set2d t x y     v
  [x, y, z]    -> _set3d t x y z   v
  [x, y, z, q] -> _set4d t x y z q v
  _            -> throwGT4 "set"

-- setDim'_ :: (Dimensions d) => Tensor d -> SomeDims -> HsReal (Tensor d) -> IO ()
-- setDim'_ t (SomeDims d) v = _setDim t d v

getDim :: Dimensions d' => Tensor d -> Dim (d'::[Nat]) -> IO (HsReal)
getDim t d = case dimVals d of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> throwGT4 "get"

getDims :: Tensor d -> IO SomeDims
getDims t = do
  nd <- nDimension t
  ds <- mapM (size t . fromIntegral) [0 .. nd -1]
  someDimsM ds


-- | select a dimension of a tensor. If a vector is passed in, return a singleton tensor
-- with the index value of the vector.
(!!)
  :: forall t (d::[Nat]) (d'::[Nat])
  .  (TensorCopy t, HsReal (t d) ~ HsReal (t d'), Dimensions2 d d', IsTensor t)
  => t d -> DimVal -> t d'
t !! i = unsafePerformIO $
  nDimension t >>= \case
    0 -> empty
    1 -> runMaybeT selectVal >>= maybe empty pure
    _ -> selectRank

  where
    selectVal :: MaybeT IO (t d')
    selectVal = do
      sizeI <- fromIntegral <$> lift (size t i)
      guard (i < sizeI)
      v <- lift $ get1d t (fromIntegral i)
      r <- lift $ newWithSize1d 1
      lift $ _set1d r 0 v
      pure r

    selectRank :: IO (t d')
    selectRank = do
      sz <- fmap fromIntegral (size t i)
      r <- newSelect r' i 0
      pure (r :: t d')


new :: forall d . Dimensions d => IO (Tensor d)
new = case dimVals d of
  []           -> empty
  [x]          -> newWithSize1d x
  [x, y]       -> newWithSize2d x y
  [x, y, z]    -> newWithSize3d x y z
  [x, y, z, q] -> newWithSize4d x y z q
  _ -> empty >>= _resizeDim
 where
  d :: Dim d
  d = dim

-- NOTE: This is copied from the dynamic version to keep the constraints clean and is _unsafe_
_resizeDim :: forall d d' . (Dimensions d') => Tensor d -> IO (Tensor d')
_resizeDim t = case dimVals d of
  []              -> throwNE "can't resize to an empty dimension."
  [x]             -> _resize1d t x
  [x, y]          -> _resize2d t x y
  [x, y, z]       -> _resize3d t x y z
  [x, y, z, q]    -> _resize4d t x y z q
  [x, y, z, q, w] -> _resize5d t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
 where
  d :: Dim d'
  d = dim
  -- ds              -> _resizeNd t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")

view :: forall d d' . (Dimensions d, Dimensions d') => Tensor d -> IO (Tensor d')
view src = do
  res <- newClone src
  shape :: Tensor d' <- new
  _resizeAs res shape

resizeAs :: forall t d d' . (Dimensions2 d d', IsTensor t, Product d ~ Product d') => t d -> IO (t d')
resizeAs src = do
  shape <- new
  _resizeAs src shape

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME: There might be a faster way to do this with newWithSize
fromList
  :: forall d .  Dimensions d
  => [HsReal] -> Maybe (Tensor d)
fromList l = unsafePerformIO . runMaybeT $ do
  vec :: Tensor '[Product d] <- MaybeT (pure (vector l))
  guard (genericLength l == natVal (Proxy :: Proxy (Product d)))
  lift $ _resizeDim vec
{-# NOINLINE fromList #-}

matrix
  :: forall n m
  .  (KnownNatDim3 n m (n*m))
  => [[HsReal]] -> Either String (Tensor '[n, m])
matrix ls
  | length ls == 0 = Left "no support for empty lists"
  | genericLength ls /= (natVal (Proxy :: Proxy n)) = Left "length of outer list must match type-level columns"
  | any (/= length (head ls)) (fmap length ls) = Left "can't build a matrix from jagged lists"
  | genericLength (head ls) /= (natVal (Proxy :: Proxy n)) = Left "inner list length must match type-level rows"
  | otherwise =
    case fromList (concat ls) of
      Nothing -> Left "impossible: number of elements doesn't match the dimensions"
      Just m -> Right m


newTranspose2d :: (KnownNat2 r c) => Tensor '[r, c] -> IO (Tensor '[c, r])
newTranspose2d t = newTranspose t 1 0

-- | Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
expand2d
  :: forall x y . (KnownNatDim2 x y)
  => Tensor '[x] -> IO (Tensor '[y, x])
expand2d t = do
  res :: Tensor '[y, x] <- new
  s <- mkLongStorage =<< TH.c_newWithSize2_ s2 s1
  _expand res t s
  pure res
  where
    s1 = fromIntegral $ natVal (Proxy :: Proxy x)
    s2 = fromIntegral $ natVal (Proxy :: Proxy y)

getElem2d
  :: forall n m . (KnownNatDim2 n m)
  => Tensor '[n, m] -> Natural -> Natural -> IO (HsReal)
getElem2d t r c
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = get2d t (fromIntegral r) (fromIntegral c)

setElem2d
  :: forall n m ns . (KnownNatDim2 n m)
  => Tensor '[n, m] -> Natural -> Natural -> HsReal -> IO ()
setElem2d t r c v
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = _set2d t (fromIntegral r) (fromIntegral c) v


