-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE CPP #-}
module Torch.Indef.Static.Tensor
  ( module X
  , _expandNd
  , tensordata
  , fromList
  , scalar
  , vector, unsafeVector
  , matrix, unsafeMatrix
  , cuboid, unsafeCuboid
  ) where

import Torch.Indef.Static.Tensor.Internal as X

import Control.Monad
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except
import Control.Monad.Trans.Maybe
import Data.List (genericLength)
import Data.Singletons.Prelude.List hiding (All, type (++))
import GHC.TypeLits
import Numeric.Dimensions
import System.IO.Unsafe (unsafeDupablePerformIO)

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor as Dynamic

-- | Static call to 'Dynamic._expandNd'
_expandNd rs os = Dynamic._expandNd (fmap asDynamic rs) (fmap asDynamic os)

-- | Static call to 'Dynamic.tensordata'
tensordata t = Dynamic.tensordata (asDynamic t)

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
{-# NOINLINE unsafeVector #-}


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


