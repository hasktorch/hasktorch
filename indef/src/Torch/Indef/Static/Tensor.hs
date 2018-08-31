-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fno-cse #-} -- -fplugin GHC.TypeLits.Normalise #-}
module Torch.Indef.Static.Tensor where

import Control.Exception.Safe
import Control.Monad
import Control.Monad.Trans
import Control.Monad.Trans.Maybe
import Data.Coerce
import Data.Maybe
import Data.List
import Data.Singletons.Prelude.List hiding (All, type (++))
import Data.Proxy
import GHC.Natural
import System.IO.Unsafe
import GHC.TypeLits
import Numeric.Dimensions

import Torch.Indef.Types
import Torch.Indef.Index
import Torch.Indef.Static.Tensor.Copy
import qualified Torch.Indef.Dynamic.Tensor as Dynamic
import qualified Torch.Types.TH as TH
import qualified Torch.FFI.TH.Long.Storage as TH
import qualified Torch.Sig.Types as Sig

instance Dimensions d => Show (Tensor d) where
  show t = unsafePerformIO $ do
    (vs, desc) <-
      Dynamic.showTensor -- (pure . get1d t) (get2d t) (get3d t) (get4d t) (fromIntegral <$> listDims (dims :: Dims d))
        (pure . get1d t) (pure .: get2d t) (\a b c -> pure $ get3d t a b c) (\a b c d -> pure $ get4d t a b c d)
        (fromIntegral <$> listDims (dims :: Dims d))
    pure (vs ++ "\n" ++ desc)
  {-# NOINLINE show #-}

-- unnessecary
-- -- | same as 'Dynamic.isSameSizeAs' but only uses type-level dimensions to compute.
-- isSameSizeAs :: forall d d' . (All Dimensions '[d, d']) => Tensor d -> Tensor d' -> Bool
-- isSameSizeAs _ _ = (fromIntegral <$> listDims (dim :: Dim d)) == (fromIntegral <$> listDims (dim :: Dim d'))

scalar :: HsReal -> Tensor '[1]
scalar = unsafeVector . (:[])

-- | Purely make a 1d tensor from a list of unknown length.
vector :: forall n . KnownDim n => KnownNat n => [HsReal] -> Either String (Tensor '[n])
vector rs
  | genericLength rs == dimVal (dim :: Dim n) = Right . asStatic . Dynamic.vector $ rs
  | otherwise = Left "Vector dimension does not match length of list"

unsafeVector :: (KnownDim n, KnownNat n) => [HsReal] -> Tensor '[n]
unsafeVector = either error id  . vector

-- | Static call to 'Dynamic.newExpand'
newExpand t = fmap asStatic . Dynamic.newExpand (asDynamic t)
-- | Static call to 'Dynamic._expand'
_expand r t = Dynamic._expand (asDynamic r) (asDynamic t)
-- | Static call to 'Dynamic._expandNd'
_expandNd rs os = Dynamic._expandNd (fmap asDynamic rs) (fmap asDynamic os)

-- | Static call to 'Dynamic._resize'
_resize t a b = Dynamic._resize (asDynamic t) a b >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic._resize1d'
_resize1d t a = Dynamic._resize1d (asDynamic t) a >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic._resize2d'
_resize2d t a b = Dynamic._resize2d (asDynamic t) a b >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic._resize3d'
_resize3d t a b c = Dynamic._resize3d (asDynamic t) a b c >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic._resize4d'
_resize4d t a b c d = Dynamic._resize4d (asDynamic t) a b c d >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic._resize5d'
_resize5d t a b c d e = Dynamic._resize5d (asDynamic t) a b c d e >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic._resizeAs'
_resizeAs src tar = Dynamic._resizeAs (asDynamic src) (asDynamic tar) >> pure ((asStatic . asDynamic) src)
-- | Static call to 'Dynamic._resizeNd'
_resizeNd src a b c = Dynamic._resizeNd (asDynamic src) a b c >> pure ((asStatic . asDynamic) src)
-- | Static call to 'Dynamic.retain'
retain t = Dynamic.retain (asDynamic t)
-- | Static call to 'Dynamic._clearFlag'
_clearFlag t = Dynamic._clearFlag (asDynamic t)
-- | Static call to 'Dynamic.tensordata'
tensordata t = Dynamic.tensordata (asDynamic t)
-- | Static call to 'Dynamic.get1d'
get1d t = Dynamic.get1d (asDynamic t)
-- | Static call to 'Dynamic.get2d'
get2d t = Dynamic.get2d (asDynamic t)
-- | Static call to 'Dynamic.get3d'
get3d t = Dynamic.get3d (asDynamic t)
-- | Static call to 'Dynamic.get4d'
get4d t = Dynamic.get4d (asDynamic t)
-- | Static call to 'Dynamic.isContiguous'
isContiguous t = Dynamic.isContiguous (asDynamic t)

-- | Static call to 'Dynamic.isSetTo'
isSetTo t0 t1 = Dynamic.isSetTo (asDynamic t0) (asDynamic t1)
-- | Static call to 'Dynamic.isSize'
isSize t = Dynamic.isSize (asDynamic t)
-- | Static call to 'Dynamic.nDimension'
nDimension t = Dynamic.nDimension (asDynamic t)
-- | Static call to 'Dynamic.nElement'
nElement t = Dynamic.nElement (asDynamic t)
-- | Static call to 'Dynamic._narrow'
_narrow t0 t1 = Dynamic._narrow (asDynamic t0) (asDynamic t1)

-- | renamed from TH's @new@ because this always returns an empty tensor
-- FIXME: this __technically should be @IO (Tensor '[])@, but if you leave it as-is
-- the types line-up nicely (and we currently don't use rank-0 tensors).
empty = asStatic <$> Dynamic.empty

-- | Static call to 'Dynamic.newClone'
newClone t = asStatic <$> Dynamic.newClone (asDynamic t)
-- | Static call to 'Dynamic.newContiguous'
newContiguous t = asStatic <$> Dynamic.newContiguous (asDynamic t)
-- | Static call to 'Dynamic.newNarrow'
newNarrow t a b c = asStatic <$> Dynamic.newNarrow (asDynamic t) a b c

-- | Static call to 'Dynamic.newSelect'
newSelect
  :: KnownDim i
  => '(ls, r:+rs) ~ SplitAt i d
  => Tensor d
  -> (Dim i, Idx i)
  -> IO (Tensor (ls ++ rs))
newSelect t (d, i) =
  asStatic <$>
    Dynamic.newSelect
      (asDynamic t)
      (fromIntegral (dimVal d))
      (fromIntegral (fromEnum i))

-- | Static call to 'Dynamic.newSizeOf'
newSizeOf t = Dynamic.newSizeOf (asDynamic t)
-- | Static call to 'Dynamic.newStrideOf'
newStrideOf t = Dynamic.newStrideOf (asDynamic t)
-- | Static call to 'Dynamic.newTranspose'
newTranspose t a b = asStatic <$> Dynamic.newTranspose (asDynamic t) a b
-- | Static call to 'Dynamic.newUnfold'
newUnfold t a b c = asStatic <$> Dynamic.newUnfold (asDynamic t) a b c

-- | Make a new view of a tensor.
view :: forall d d' . (Dimensions d, Dimensions d') => Tensor d -> IO (Tensor d')
view src = do
  longs <- ixCPUStorage $ fromIntegral <$> listDims (dims :: Dims d)
  asStatic <$> Dynamic.newView (asDynamic src) longs


-- | Static call to 'Dynamic.newWithSize'
newWithSize a0 a1 = asStatic <$> Dynamic.newWithSize a0 a1
-- | Static call to 'Dynamic.newWithSize1d'
newWithSize1d a0 = asStatic <$> Dynamic.newWithSize1d a0
-- | Static call to 'Dynamic.newWithSize2d'
newWithSize2d a0 a1 = asStatic <$> Dynamic.newWithSize2d a0 a1
-- | Static call to 'Dynamic.newWithSize3d'
newWithSize3d a0 a1 a2 = asStatic <$> Dynamic.newWithSize3d a0 a1 a2
-- | Static call to 'Dynamic.newWithSize4d'
newWithSize4d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithSize4d a0 a1 a2 a3
-- | Static call to 'Dynamic.newWithStorage'
newWithStorage a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage a0 a1 a2 a3
-- | Static call to 'Dynamic.newWithStorage1d'
newWithStorage1d a0 a1 a2 = asStatic <$> Dynamic.newWithStorage1d a0 a1 a2
-- | Static call to 'Dynamic.newWithStorage2d'
newWithStorage2d a0 a1 a2 a3 = asStatic <$> Dynamic.newWithStorage2d a0 a1 a2 a3
-- | Static call to 'Dynamic.newWithStorage3d'
newWithStorage3d a0 a1 a2 a3 a4 = asStatic <$> Dynamic.newWithStorage3d a0 a1 a2 a3 a4
-- | Static call to 'Dynamic.newWithStorage4d'
newWithStorage4d a0 a1 a2 a3 a4 a5 = asStatic <$> Dynamic.newWithStorage4d a0 a1 a2 a3 a4 a5
-- | Static call to 'Dynamic.newWithTensor'
newWithTensor t = asStatic <$> Dynamic.newWithTensor (asDynamic t)
-- | Static call to 'Dynamic._select'
_select t0 t1 = Dynamic._select (asDynamic t0) (asDynamic t1)
-- | Static call to 'Dynamic._set'
_set t0 t1 = Dynamic._set (asDynamic t0) (asDynamic t1)
-- | Static call to 'Dynamic._set1d'
_set1d t = Dynamic._set1d (asDynamic t)
-- | Static call to 'Dynamic._set2d'
_set2d t = Dynamic._set2d (asDynamic t)
-- | Static call to 'Dynamic._set3d'
_set3d t = Dynamic._set3d (asDynamic t)
-- | Static call to 'Dynamic._set4d'
_set4d t = Dynamic._set4d (asDynamic t)
-- | Static call to 'Dynamic._setFlag'
_setFlag t = Dynamic._setFlag (asDynamic t)
-- | Static call to 'Dynamic._setStorage'
_setStorage t = Dynamic._setStorage (asDynamic t)
-- | Static call to 'Dynamic._setStorage1d'
_setStorage1d t = Dynamic._setStorage1d (asDynamic t)
-- | Static call to 'Dynamic._setStorage2d'
_setStorage2d t = Dynamic._setStorage2d (asDynamic t)
-- | Static call to 'Dynamic._setStorage3d'
_setStorage3d t = Dynamic._setStorage3d (asDynamic t)
-- | Static call to 'Dynamic._setStorage4d'
_setStorage4d t = Dynamic._setStorage4d (asDynamic t)
-- | Static call to 'Dynamic._setStorageNd'
_setStorageNd t = Dynamic._setStorageNd (asDynamic t)
-- | Static call to 'Dynamic.size'
size t = Dynamic.size (asDynamic t)
-- | Static call to 'Dynamic.sizeDesc'
sizeDesc t = Dynamic.sizeDesc (asDynamic t)

-- | Static call to 'Dynamic._squeeze'
_squeeze t0 t1 = Dynamic._squeeze (asDynamic t0) (asDynamic t1)

-- | Squeeze a dimension of size 1 out of the tensor
squeeze1d
  :: Dimensions d
  => '(rs, 1:+ls) ~ (SplitAt n d)
  => Dim n
  -> Tensor d
  -> Tensor (rs ++ ls)
squeeze1d n t = unsafeDupablePerformIO $ do
  let t' = (asStatic . asDynamic) (copy t)
  Dynamic._squeeze1d (asDynamic t') (asDynamic t) (fromIntegral (dimVal n))
  pure t'

-- | Static call to 'Dynamic.storage'
storage t = Dynamic.storage (asDynamic t)
-- | Static call to 'Dynamic.storageOffset'
storageOffset t = Dynamic.storageOffset (asDynamic t)
-- | Static call to 'Dynamic.stride'
stride t = Dynamic.stride (asDynamic t)
-- | Static call to 'Dynamic._transpose'
_transpose t0 t1 = Dynamic._transpose (asDynamic t0) (asDynamic t1)
-- | Static call to 'Dynamic._unfold'
_unfold t0 t1 = Dynamic._unfold (asDynamic t0) (asDynamic t1)

-- | Unsqueeze a dimension of size 1 into the tensor (pure, dupable)
unsqueeze1d
  :: Dimensions d
  => '(rs, ls) ~ (SplitAt n d)
  => Dim n
  -> Tensor d
  -> Tensor (rs ++ '[1] ++ ls)
unsqueeze1d n t = unsafeDupablePerformIO $ do
  let t' = (asStatic . asDynamic) (copy t)
  Dynamic._unsqueeze1d (asDynamic t') (asDynamic t) (fromIntegral (dimVal n))
  pure t'

-- | *Not safe:*  unsqueeze a dimension of size 1 into the tensor.
unsqueeze1d_
  :: Dimensions d
  => '(rs, ls) ~ (SplitAt n d)
  => Dim n
  -> Tensor d
  -> IO (Tensor (rs ++ '[1] ++ ls))
unsqueeze1d_ n t = do
  Dynamic._unsqueeze1d (asDynamic t) (asDynamic t) (fromIntegral (dimVal n))
  pure (asStatic (asDynamic t))


-- ========================================================================= --

-- | Get runtime shape information from a tensor
shape :: Tensor d -> [Word]
shape t = Dynamic.shape (asDynamic t)

-- | alias to 'shape', casting the dimensions into a runtime 'SomeDims'.
getDims :: Tensor d -> SomeDims
getDims = someDimsVal . shape

-- | helper function to make other parts of hasktorch valid pure functions.
withNew :: forall d . (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
withNew op = new >>= \r -> op r >> pure r

-- | Should be renamed to @newFromSize@
withEmpty :: forall d . (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
withEmpty op = new >>= \r -> op r >> pure r

-- | same as 'withEmpty' (which should be called @newFromSize@) and 'withNew',
-- but passes in an empty tensor to be mutated and returned with a static
-- dimensionality that it is assumed to take on after the mutation.
--
-- Note: We can get away with this when Torch does resizing in C, but you need
-- to examine the C implementation of the function you are trying to make pure.
withEmpty' :: (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
withEmpty' op = empty >>= \r -> op r >> pure r

-- |
-- This is actually 'inplace'. Dimensions may change from original tensor given Torch resizing.
--
-- FIXME: remove this function
withInplace :: (Dimensions d) => Tensor d -> (Tensor d -> Tensor d -> IO ()) -> IO (Tensor d)
withInplace t op = op t t >> pure t
{-# DEPRECATED withInplace "this is a trivial function with a bad API" #-}

-- | throw a "FIXME" string.
throwFIXME :: MonadThrow io => String -> String -> io x
throwFIXME fixme msg = throwString $ msg ++ " (FIXME: " ++ fixme ++ ")"

-- | throw an "unsafe head" string.
throwNE :: MonadThrow io => String -> io x
throwNE = throwFIXME "make this function only take a non-empty [Nat]"

-- | throw an "unsupported dimension" string.
throwGT4 :: MonadThrow io => String -> io x
throwGT4 fnname = throwFIXME
  ("review how TH supports `" ++ fnname ++ "` operations on > rank-4 tensors")
  (fnname ++ " with >4 rank")


-- | Set the storage of a tensor. This is incredibly unsafe.
_setStorageDim :: Tensor d -> Storage -> StorageOffset -> [(Size, Stride)] -> IO ()
_setStorageDim t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> _setStorage1d t s o x
  [x, y]       -> _setStorage2d t s o x y
  [x, y, z]    -> _setStorage3d t s o x y z
  [x, y, z, q] -> _setStorage4d t s o x y z q
  _            -> throwGT4 "setStorage"

-- | Set the value of a tensor at a given index
--
-- FIXME: there should be a constraint to see that d' is in d
setDim_ :: Tensor d -> Dims (d'::[Nat]) -> HsReal -> IO ()
setDim_ t d v = case fromIntegral <$> listDims d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> _set1d t x       v
  [x, y]       -> _set2d t x y     v
  [x, y, z]    -> _set3d t x y z   v
  [x, y, z, q] -> _set4d t x y z q v
  _            -> throwGT4 "set"


-- | runtime version of 'setDim_'
setDim'_ :: Tensor d -> SomeDims -> HsReal -> IO ()
setDim'_ t (SomeDims d) = setDim_ t d -- (d :: Dims d')

-- | get the value of a tensor at the given index
--
-- FIXME: there should be a constraint to see that d' is in d
getDim :: forall d d' . All Dimensions '[d, d'] => Tensor (d::[Nat]) -> Dims (d'::[Nat]) -> IO (HsReal)
getDim t d = case fromIntegral <$> listDims (dims :: Dims d) of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> pure $ get1d t x
  [x, y]       -> pure $ get2d t x y
  [x, y, z]    -> pure $ get3d t x y z
  [x, y, z, q] -> pure $ get4d t x y z q
  _            -> throwGT4 "get"

-- | Select a dimension of a tensor. If a vector is passed in, return a singleton tensor
-- with the index value of the vector.
(!!)
  :: forall d ls r rs i
  .  '(ls, r:+rs) ~ SplitAt i d
  => KnownDim i
  => Dimensions d
  => Tensor d
  -> Dim i
  -> Tensor (ls ++ rs)
t !! i = unsafePerformIO $
  case nDimension t of
    0 -> empty
    1 -> runMaybeT selectVal >>= maybe empty pure
    _ -> newSelect t (i, Idx 1)

  where
    selectVal :: MaybeT IO (Tensor (ls ++ rs))
    selectVal = do
      sizeI <- fromIntegral <$> lift (size t (fromIntegral $ dimVal i))
      guard (dimVal i < sizeI)
      r <- lift $ newWithSize1d 1
      lift $ _set1d r 0 (get1d t (fromIntegral $ dimVal i))
      pure r

-- | Create a new tensor. Elements have not yet been allocated and there will
-- not be any gauruntees regarding what is inside.
new :: forall d . Dimensions d => IO (Tensor d)
new = asStatic <$> Dynamic.new (dims :: Dims d)

-- | Resize a tensor, returning the same tensor with a the changed dimensions.
--
-- NOTE: This is copied from the dynamic version to keep the constraints clean and is _unsafe_
_resizeDim :: forall d d' . (Dimensions d') => Tensor d -> IO (Tensor d')
_resizeDim t = case fromIntegral <$> listDims (dims :: Dims d') of
  []              -> throwNE "can't resize to an empty dimension."
  [x]             -> _resize1d t x
  [x, y]          -> _resize2d t x y
  [x, y, z]       -> _resize3d t x y z
  [x, y, z, q]    -> _resize4d t x y z q
  [x, y, z, q, w] -> _resize5d t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
  -- ds              -> _resizeNd t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")

-- | Resize the input with the output shape. impure and mutates the tensor inplace.
--
-- FIXME: replace @d@ with a linear type?
resizeAs_ :: forall d d' . (All Dimensions '[d, d'], Product d ~ Product d') => Tensor d -> IO (Tensor d')
resizeAs_ src = do
  shape :: Tensor d' <- new
  _resizeAs src shape

-- | Pure version of 'resizeAs_' which clones the input tensor (pure, dupable?)
resizeAs :: forall d d' . (All Dimensions [d,d'], Product d ~ Product d') => Tensor d -> Tensor d'
resizeAs src = unsafeDupablePerformIO $ do
  res <- newClone src
  shape :: Tensor d' <- new
  _resizeAs res shape

-- | flatten a tensor (pure, dupable)
flatten :: (Dimensions d, KnownDim (Product d)) => Tensor d -> Tensor '[Product d]
flatten = resizeAs

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME: There might be a faster way to do this with newWithSize
fromList
  :: forall d .  Dimensions d
  => KnownNat (Product d)
  => KnownDim (Product d)
  => [HsReal] -> Maybe (Tensor d)
fromList l = unsafePerformIO . runMaybeT $ do
  vec :: Tensor '[Product d] <-
    case vector l of
      Left _ -> mzero
      Right t -> pure t
  guard (genericLength l == dimVal (dim :: Dim (Product d)))
  lift $ _resizeDim vec
{-# NOINLINE fromList #-}

-- | Purely construct a matrix from a list of lists. This assumes that the list of lists is
-- a dense matrix representation. Returns either the successful construction of the tensor,
-- or a string explaining what went wrong.
matrix
  :: forall n m
  .  (All KnownDim '[n, m], All KnownNat '[n, m])
#if MIN_VERSION_singletons(2,4,0)
  => KnownDim (n*m) => KnownNat (n*m)
#else
  => KnownDim (n*:m) => KnownNat (n*:m)
#endif
  => [[HsReal]] -> Either String (Tensor '[n, m])
matrix ls
  | null ls = Left "no support for empty lists"

  | colLen /= mVal =
    Left $ "length of outer list "++show colLen++" must match type-level columns " ++ show mVal

  | any (/= colLen) (fmap length ls) =
    Left $ "can't build a matrix from jagged lists: " ++ show (fmap length ls)

  | rowLen /= nVal =
    Left $ "inner list length " ++ show rowLen ++ " must match type-level rows " ++ show nVal

  | otherwise =
    case fromList (concat ls) of
      Nothing -> Left "impossible: number of elements doesn't match the dimensions"
      Just m -> Right m
  where
    rowLen :: Integral l => l
    rowLen = genericLength ls

    colLen :: Integral l => l
    colLen = genericLength (head ls)

    nVal = dimVal (dim :: Dim n)
    mVal = dimVal (dim :: Dim m)

unsafeMatrix
  :: forall n m
  .  All KnownDim '[n, m, n*m]
  => All KnownNat '[n, m, n*m]
  => [[HsReal]] -> Tensor '[n, m]
unsafeMatrix = either error id . matrix

-- | transpose a matrix (pure, dupable)
transpose2d :: (All KnownDim '[r,c]) => Tensor '[r, c] -> Tensor '[c, r]
transpose2d t = unsafeDupablePerformIO $ newTranspose t 1 0

-- | Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
expand2d
  :: forall x y . (All KnownDim '[x, y])
  => Tensor '[x] -> Tensor '[y, x]
expand2d t = unsafeDupablePerformIO $ do
  res :: Tensor '[y, x] <- new
  s <- mkCPUIxStorage =<< TH.c_newWithSize2_ s2 s1
  _expand res t s
  pure res
  where
    s1 = fromIntegral $ dimVal (dim :: Dim  x)
    s2 = fromIntegral $ dimVal (dim :: Dim  y)

-- | Get an element from a matrix with runtime index values.
--
-- FIXME: This is primarily for backwards compatability with lasso
-- and should be removed.
getElem2d
  :: forall (n::Nat) (m::Nat) . (All KnownDim '[n, m])
  => Tensor '[n, m] -> Natural -> Natural -> IO (HsReal)
getElem2d t r c
  | r > fromIntegral (dimVal (dim :: Dim n)) ||
    c > fromIntegral (dimVal (dim :: Dim m))
      = throwString "Indices out of bounds"
  | otherwise = pure $ get2d t (fromIntegral r) (fromIntegral c)
{-# DEPRECATED getElem2d "use getDim instead" #-}

-- | Set an element on a matrix with runtime index values.
--
-- FIXME: This is primarily for backwards compatability with lasso
-- and should be removed.
setElem2d
  :: forall (n::Nat) (m::Nat) ns . (All KnownDim '[n, m])
  => Tensor '[n, m] -> Natural -> Natural -> HsReal -> IO ()
setElem2d t r c v
  | r > fromIntegral (dimVal (dim :: Dim n)) ||
    c > fromIntegral (dimVal (dim :: Dim m))
      = throwString "Indices out of bounds"
  | otherwise = _set2d t (fromIntegral r) (fromIntegral c) v
{-# DEPRECATED setElem2d "use setDim_ instead" #-}


