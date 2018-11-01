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

#if MIN_VERSION_base(4,12,0)
{-# LANGUAGE NoStarIsType #-}
#endif

{-# OPTIONS_GHC -fno-cse -Wno-deprecations #-} -- no deprications because we still bundle up all mutable functions
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
import Control.Monad.Trans.Except

import Torch.Indef.Types
import Torch.Indef.Index
import Torch.Indef.Static.Tensor.Copy
import qualified Torch.Indef.Dynamic.Tensor as Dynamic
import qualified Torch.Types.TH as TH
import qualified Torch.FFI.TH.Long.Storage as TH
import qualified Torch.Sig.Types as Sig

instance Show (Tensor d) where
  show t = show (asDynamic t)

-- unnessecary
-- -- | same as 'Dynamic.isSameSizeAs' but only uses type-level dimensions to compute.
-- isSameSizeAs :: forall d d' . (All Dimensions '[d, d']) => Tensor d -> Tensor d' -> Bool
-- isSameSizeAs _ _ = (fromIntegral <$> listDims (dim :: Dim d)) == (fromIntegral <$> listDims (dim :: Dim d'))

scalar :: HsReal -> Tensor '[1]
scalar = unsafeDupablePerformIO . unsafeVector . (:[])
{-# NOINLINE scalar #-}

-- | Purely make a 1d tensor from a list of unknown length.
vector :: forall n . KnownDim n => KnownNat n => [HsReal] -> ExceptT String IO (Tensor '[n])
vector rs
  | genericLength rs == dimVal (dim :: Dim n) = asStatic <$> Dynamic.vectorEIO rs
  | otherwise = ExceptT . pure $ Left "Vector dimension does not match length of list"

unsafeVector :: (KnownDim n, KnownNat n) => [HsReal] -> IO (Tensor '[n])
unsafeVector = fmap (either error id) . runExceptT . vector

-- | Static call to 'Dynamic.newExpand'
newExpand :: Tensor d -> TH.IndexStorage -> Tensor d'
newExpand t = asStatic . Dynamic.newExpand (asDynamic t)

-- | Static call to 'Dynamic._expand'
_expand r t = Dynamic._expand (asDynamic r) (asDynamic t)
-- | Static call to 'Dynamic._expandNd'
_expandNd rs os = Dynamic._expandNd (fmap asDynamic rs) (fmap asDynamic os)

-- | Static call to 'Dynamic.resize_'
_resize t a b = Dynamic._resize (asDynamic t) a b >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic.resize1d_'
resize1d_ t a = Dynamic.resize1d_ (asDynamic t) a >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic.resize2d_'
resize2d_ t a b = Dynamic.resize2d_ (asDynamic t) a b >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic.resize3d_'
resize3d_ t a b c = Dynamic.resize3d_ (asDynamic t) a b c >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic.resize4d_'
resize4d_ t a b c d = Dynamic.resize4d_ (asDynamic t) a b c d >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic.resize5d_'
resize5d_ t a b c d e = Dynamic.resize5d_ (asDynamic t) a b c d e >> pure ((asStatic . asDynamic) t)
-- | Static call to 'Dynamic.resizeAs_'
resizeAsT_ src tar = Dynamic.resizeAs_ (asDynamic src) (asDynamic tar) >> pure ((asStatic . asDynamic) src)
-- | Static call to 'Dynamic.resizeNd_'
resizeNd_ src a b c = Dynamic.resizeNd_ (asDynamic src) a b c >> pure ((asStatic . asDynamic) src)
-- | Static call to 'Dynamic.retain'
retain t = Dynamic.retain (asDynamic t)
-- | Static call to 'Dynamic._clearFlag'
_clearFlag t = Dynamic._clearFlag (asDynamic t)
#ifndef HASKTORCH_CORE_CUDA
-- | Static call to 'Dynamic.tensordata'
tensordata t = Dynamic.tensordata (asDynamic t)
#endif
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
empty = asStatic Dynamic.empty

-- | Static call to 'Dynamic.newClone'
newClone :: Tensor d -> Tensor d
newClone t = asStatic $ Dynamic.newClone (asDynamic t)

-- | Static call to 'Dynamic.newContiguous'
newContiguous t = asStatic $ Dynamic.newContiguous (asDynamic t)
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
newTranspose t a b = asStatic $ Dynamic.newTranspose (asDynamic t) a b

-- | Static call to 'Dynamic.newUnfold'
newUnfold t a b c = asStatic $ Dynamic.newUnfold (asDynamic t) a b c

-- | Make a new view of a tensor.
view :: forall d d' . (Dimensions d, Dimensions d') => Tensor d -> IO (Tensor d')
view src = do
  longs <- ixCPUStorage $ fromIntegral <$> listDims (dims :: Dims d)
  asStatic <$> Dynamic.newView (asDynamic src) longs


-- | Static call to 'Dynamic.newWithSize'
newWithSize a0 a1 = asStatic $ Dynamic.newWithSize a0 a1
-- | Static call to 'Dynamic.newWithSize1d'
newWithSize1d a0 = asStatic $ Dynamic.newWithSize1d a0
-- | Static call to 'Dynamic.newWithSize2d'
newWithSize2d a0 a1 = asStatic $ Dynamic.newWithSize2d a0 a1
-- | Static call to 'Dynamic.newWithSize3d'
newWithSize3d a0 a1 a2 = asStatic $ Dynamic.newWithSize3d a0 a1 a2
-- | Static call to 'Dynamic.newWithSize4d'
newWithSize4d a0 a1 a2 a3 = asStatic $ Dynamic.newWithSize4d a0 a1 a2 a3
-- | Static call to 'Dynamic.newWithStorage'
newWithStorage a0 a1 a2 a3 = asStatic $ Dynamic.newWithStorage a0 a1 a2 a3
-- | Static call to 'Dynamic.newWithStorage1d'
newWithStorage1d a0 a1 a2 = asStatic $ Dynamic.newWithStorage1d a0 a1 a2
-- | Static call to 'Dynamic.newWithStorage2d'
newWithStorage2d a0 a1 a2 a3 = asStatic $ Dynamic.newWithStorage2d a0 a1 a2 a3
-- | Static call to 'Dynamic.newWithStorage3d'
newWithStorage3d a0 a1 a2 a3 a4 = asStatic $ Dynamic.newWithStorage3d a0 a1 a2 a3 a4
-- | Static call to 'Dynamic.newWithStorage4d'
newWithStorage4d a0 a1 a2 a3 a4 a5 = asStatic $ Dynamic.newWithStorage4d a0 a1 a2 a3 a4 a5
-- | Static call to 'Dynamic.newWithTensor'
newWithTensor t = asStatic <$> Dynamic.newWithTensor (asDynamic t)
-- | Static call to 'Dynamic._select'
_select t0 t1 = Dynamic._select (asDynamic t0) (asDynamic t1)
-- | Static call to 'Dynamic._set'
_set t0 t1 = Dynamic._set (asDynamic t0) (asDynamic t1)
-- | Static call to 'Dynamic.set1d_'
set1d_ t = Dynamic.set1d_ (asDynamic t)
-- | Static call to 'Dynamic.set2d_'
set2d_ t = Dynamic.set2d_ (asDynamic t)
-- | Static call to 'Dynamic.set3d_'
set3d_ t = Dynamic.set3d_ (asDynamic t)
-- | Static call to 'Dynamic.set4d_'
set4d_ t = Dynamic.set4d_ (asDynamic t)
-- | Static call to 'Dynamic.setFlag_'
setFlag_ t = Dynamic.setFlag_ (asDynamic t)
-- | Static call to 'Dynamic.setStorage_'
setStorage_ t = Dynamic.setStorage_ (asDynamic t)
-- | Static call to 'Dynamic.setStorage1d_'
setStorage1d_ t = Dynamic.setStorage1d_ (asDynamic t)
-- | Static call to 'Dynamic.setStorage2d_'
setStorage2d_ t = Dynamic.setStorage2d_ (asDynamic t)
-- | Static call to 'Dynamic.setStorage3d_'
setStorage3d_ t = Dynamic.setStorage3d_ (asDynamic t)
-- | Static call to 'Dynamic.setStorage4d_'
setStorage4d_ t = Dynamic.setStorage4d_ (asDynamic t)
-- | Static call to 'Dynamic.setStorageNd_'
setStorageNd_ t = Dynamic.setStorageNd_ (asDynamic t)
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
squeeze1d n t =  unsafeDupablePerformIO $ squeeze1d_ n (newClone t)
{-# NOINLINE squeeze1d #-}

-- | *Not safe:*  squeeze a dimension of size 1 out of the tensor.
squeeze1d_
  :: Dimensions d
  => '(rs, 1:+ls) ~ (SplitAt n d)
  => Dim n
  -> Tensor d
  -> IO (Tensor (rs ++ ls))
squeeze1d_ n t = do
  let t' = asDynamic t
  Dynamic.squeeze1d_ t' (fromIntegral (dimVal n))
  pure (asStatic t')


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
unsqueeze1d n t = unsafeDupablePerformIO $ unsqueeze1d_ n (newClone t)
{-# NOINLINE unsqueeze1d #-}

-- | *Not safe:*  unsqueeze a dimension of size 1 into the tensor.
unsqueeze1d_
  :: Dimensions d
  => '(rs, ls) ~ (SplitAt n d)
  => Dim n
  -> Tensor d
  -> IO (Tensor (rs ++ '[1] ++ ls))
unsqueeze1d_ n t = do
  Dynamic.unsqueeze1d_ (asDynamic t) (fromIntegral (dimVal n))
  pure (asStatic (asDynamic t))


-- ========================================================================= --

-- | Get runtime shape information from a tensor
shape :: Tensor d -> [Word]
shape t = Dynamic.shape (asDynamic t)

-- | alias to 'shape', casting the dimensions into a runtime 'SomeDims'.
getSomeDims :: Tensor d -> SomeDims
getSomeDims = someDimsVal . shape

-- -- | helper function to make other parts of hasktorch valid pure functions.
-- withNew :: forall d . (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
-- withNew op = new >>= \r -> op r >> pure r
--
-- -- | Should be renamed to @newFromSize@
-- withEmpty :: forall d . Dimensions d => (Tensor d -> IO ()) -> IO (Tensor d)
-- withEmpty op = let r = new in op r >> pure r

-- | same as 'withEmpty' (which should be called @newFromSize@) and 'withNew',
-- but passes in an empty tensor to be mutated and returned with a static
-- dimensionality that it is assumed to take on after the mutation.
--
-- Note: We can get away with this when Torch does resizing in C, but you need
-- to examine the C implementation of the function you are trying to make pure.
-- withEmpty' :: (Dimensions d) => (Tensor d -> IO ()) -> IO (Tensor d)
-- withEmpty' op = let r = empty in op r >> pure r

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
setStorageDim_ :: Tensor d -> Storage -> StorageOffset -> [(Size, Stride)] -> IO ()
setStorageDim_ t s o = Dynamic.setStorageDim_ (asDynamic t) s o

-- | Set the value of a tensor at a given index
--
-- FIXME: there should be a constraint to see that d' is in d
setDim_ :: Tensor d -> Dims (d'::[Nat]) -> HsReal -> IO ()
setDim_ t = Dynamic.setDim_ (asDynamic t)


-- | runtime version of 'setDim_'
setDim'_ :: Tensor d -> SomeDims -> HsReal -> IO ()
setDim'_ t (SomeDims d) = setDim_ t d -- (d :: Dims d')

-- | get the value of a tensor at the given index
--
-- FIXME: there should be a constraint to see that d' is in d
getDim
  :: forall d i d'
  .  All Dimensions '[d, i:+d']
  => Tensor (d::[Nat])
  -> Dims ((i:+d')::[Nat]) -- ^ the index to get is a non-empty dims list
  -> Maybe HsReal
getDim t d = Dynamic.getDim (asDynamic t) d

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
    0 -> pure empty
    1 -> fromMaybe empty <$> runMaybeT selectVal
    _ -> newSelect t (i, Idx 1)

  where
    selectVal :: MaybeT IO (Tensor (ls ++ rs))
    selectVal = do
      guard (dimVal i < size t (dimVal i))
      v <- MaybeT . pure $ get1d t (fromIntegral $ dimVal i)
      lift $ do
        let r = newWithSize1d 1
        set1d_ r 0 v
        pure r
{-# NOINLINE (!!) #-}


-- | Create a new tensor. Elements have not yet been allocated and there will
-- not be any gauruntees regarding what is inside.
new :: forall d . Dimensions d => Tensor d
new = asStatic $ Dynamic.new (dims :: Dims d)

-- | Resize a tensor, returning the same tensor with a the changed dimensions.
--
-- NOTE: This is copied from the dynamic version to keep the constraints clean and is _unsafe_
_resizeDim :: forall d d' . (Dimensions d') => Tensor d -> IO (Tensor d')
_resizeDim t = do
  Dynamic.resizeDim_ (asDynamic t) (dims :: Dims d')
  pure $ asStatic (asDynamic t)

-- | Resize the input with the output shape. impure and mutates the tensor inplace.
--
-- FIXME: replace @d@ with a linear type?
resizeAs_ :: forall d d' . (All Dimensions '[d, d'], Product d ~ Product d') => Tensor d -> IO (Tensor d')
resizeAs_ src = resizeAsT_ src (new :: Tensor d')

-- | Pure version of 'resizeAs_' which clones the input tensor (pure, dupable?)
--
-- WARNING: This might be not be garbage collected as you expect since the input argument becomes a dangling phantom type.
resizeAs :: forall d d' . (All Dimensions [d,d'], Product d ~ Product d') => Tensor d -> Tensor d'
resizeAs src = unsafeDupablePerformIO $
  resizeAsT_ (newClone src :: Tensor d) (new :: Tensor d')
{-# NOINLINE resizeAs #-}

-- | flatten a tensor (pure, dupable)
flatten :: (Dimensions d, KnownDim (Product d)) => Tensor d -> Tensor '[Product d]
flatten = resizeAs

-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME: There might be a faster way to do this with newWithSize
fromList
  :: forall d .  Dimensions d
  => KnownNat (Product d)
  => KnownDim (Product d)
  => [HsReal] -> IO (Maybe (Tensor d))
fromList l = runMaybeT $ do
  evec <- lift $ runExceptT (vector l)
  vec :: Tensor '[Product d] <-
    case evec of
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
  => [[HsReal]] -> ExceptT String IO (Tensor '[n, m])
matrix ls
  | null ls = ExceptT . pure . Left $ "no support for empty lists"

  | colLen /= mVal =
    ExceptT . pure . Left $ "length of outer list "++show colLen++" must match type-level columns " ++ show mVal

  | any (/= colLen) (fmap length ls) =
    ExceptT . pure . Left $ "can't build a matrix from jagged lists: " ++ show (fmap length ls)

  | rowLen /= nVal =
    ExceptT . pure . Left $ "inner list length " ++ show rowLen ++ " must match type-level rows " ++ show nVal

  | otherwise = asStatic <$> Dynamic.matrix ls
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
  => [[HsReal]] -> IO (Tensor '[n, m])
unsafeMatrix = fmap (either error id) . runExceptT . matrix

-- | Purely construct a cuboid from a list of list of lists. This assumes a dense
-- representation. Returns either the successful construction of the tensor,
-- or a string explaining what went wrong.
cuboid
  :: forall c h w
  .  (All KnownDim '[c, h, w], All KnownNat '[c, h, w])
  => [[[HsReal]]] -> ExceptT String IO (Tensor '[c, h, w])
cuboid ls
  | isEmpty ls                       = ExceptT . pure . Left $ "no support for empty lists"
  | chan /= length ls                = ExceptT . pure . Left $ "channels are not all of length " ++ show chan
  | any (/= rows) (lens ls)          = ExceptT . pure . Left $     "rows are not all of length " ++ show rows
  | any (/= cols) (lens (concat ls)) = ExceptT . pure . Left $  "columns are not all of length " ++ show cols
  | otherwise = asStatic <$> Dynamic.cuboid ls
  where
    isEmpty = \case
      []     -> True
      [[]]   -> True
      [[[]]] -> True
      list   -> null list || any null list || any (any null) list


    chan = fromIntegral $ dimVal (dim :: Dim c)
    rows = fromIntegral $ dimVal (dim :: Dim h)
    cols = fromIntegral $ dimVal (dim :: Dim w)
    lens = fmap length

    innerDimCheck :: Int -> [Int] -> Bool
    innerDimCheck d = any ((/= d))

unsafeCuboid
  :: forall c h w
  .  All KnownDim '[c, h, w]
  => All KnownNat '[c, h, w]
  => [[[HsReal]]] -> IO (Tensor '[c, h, w])
unsafeCuboid = fmap (either error id) . runExceptT . cuboid


-- | transpose a matrix (pure, dupable)
transpose2d :: (All KnownDim '[r,c]) => Tensor '[r, c] -> Tensor '[c, r]
transpose2d t = newTranspose t 1 0


-- | Expand a vector by copying into a matrix by set dimensions
-- TODO - generalize this beyond the matrix case
expand2d
  :: forall x y . (All KnownDim '[x, y])
  => Tensor '[x] -> Tensor '[y, x]
expand2d t = unsafeDupablePerformIO $ do
  let res :: Tensor '[y, x] = new
  s <- mkCPUIxStorage =<< TH.c_newWithSize2_ s2 s1
  _expand res t s
  pure res
  where
    s1 = fromIntegral $ dimVal (dim :: Dim  x)
    s2 = fromIntegral $ dimVal (dim :: Dim  y)
{-# NOINLINE expand2d #-}

-- | Get an element from a matrix with runtime index values.
--
-- FIXME: This is primarily for backwards compatability with lasso
-- and should be removed.
getElem2d
  :: forall (n::Nat) (m::Nat) . (All KnownDim '[n, m])
  => Tensor '[n, m] -> Word -> Word -> Maybe (HsReal)
getElem2d t r c
  | r > fromIntegral (dimVal (dim :: Dim n)) ||
    c > fromIntegral (dimVal (dim :: Dim m))
      = Nothing
  | otherwise = get2d t (fromIntegral r) (fromIntegral c)
{-# DEPRECATED getElem2d "use getDim instead" #-}

-- | Set an element on a matrix with runtime index values.
--
-- FIXME: This is primarily for backwards compatability with lasso
-- and should be removed.
setElem2d
  :: forall (n::Nat) (m::Nat) ns . (All KnownDim '[n, m])
  => Tensor '[n, m] -> Word -> Word -> HsReal -> IO ()
setElem2d t r c v
  | r > fromIntegral (dimVal (dim :: Dim n)) ||
    c > fromIntegral (dimVal (dim :: Dim m))
      = throwString "Indices out of bounds"
  | otherwise = set2d_ t (fromIntegral r) (fromIntegral c) v
{-# DEPRECATED setElem2d "use setDim_ instead" #-}


