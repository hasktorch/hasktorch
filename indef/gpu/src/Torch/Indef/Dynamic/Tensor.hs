-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Internal
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- This package is the class for handling numeric data in dynamic tensors. It
-- should be reexported under the Torch.Indef.Dynamic.Tensor module with
-- backend-specific code.
--
-- A 'Dynamic' is a multi-dimensional matrix without static type-level
-- dimensions. The number of dimensions is unlimited (up to what can be created
-- using LongStorage).
-------------------------------------------------------------------------------
module Torch.Indef.Dynamic.Tensor
  ( module X
  , tensordata
  , _expandNd
  , vector
  , vectorE
  , vectorEIO
  , matrix
  , hyper
  , cuboid
  ) where

import Foreign hiding (with, new)
import Foreign.Ptr
import Control.Monad
import Control.Monad.Trans.Class (lift)
import Control.Monad.Managed
import Data.List (intercalate, genericLength)
import Data.List.NonEmpty (NonEmpty(..))
import GHC.Exts (IsList(..))
import Numeric.Dimensions
import System.IO.Unsafe (unsafeDupablePerformIO, unsafePerformIO)
import Control.Monad.Trans.Except
import qualified Data.List.NonEmpty           as NE
import qualified Foreign.CUDA.Types           as CUDA
import qualified Foreign.CUDA.Runtime.Marshal as CUDA
import qualified Foreign.Marshal.Array        as FM
import qualified Torch.Sig.Types              as Sig
import qualified Torch.Sig.Tensor             as Sig

import Torch.Indef.Types
import Torch.Indef.Storage ()
import Torch.Indef.Dynamic.Tensor.Internal as X


-- | Get the underlying data as a haskell list from the tensor
--
-- NOTE: This _cannot_ use a Tensor's storage size because ATen's Storage
-- allocates up to the next 64-byte line on the CPU (needs reference, this
-- is the unofficial response from \@soumith in slack).
tensordata :: Dynamic -> [HsReal]
tensordata t =
  case shape t of
    [] -> []
    ds ->
      unsafeDupablePerformIO . flip with (pure . fmap c2hsReal) $ do
        st <- managedState
        t' <- managedTensor t
        liftIO $ do
          let sz = fromIntegral (product ds)
          -- FIXME: I think we can remove the extra copy here
          -- but I'm not going to do it just now
          tmp <- CUDA.mallocArray sz
          creals <- CUDA.DevicePtr <$> Sig.c_data st t'
          CUDA.copyArray sz tmp creals
          CUDA.peekListArray sz tmp
{-# NOINLINE tensordata #-}

----------------------------------------------------------------
-- vvvvvvvvvvvvvvvvvvv DEPENDS ON STORAGE vvvvvvvvvvvvvvvvvvv --
-- vvvvvv (specifically the IsList instance of Storage) vvvvv --
----------------------------------------------------------------

-- | FIXME: doublecheck what this does.
_expandNd  :: NonEmpty Dynamic -> NonEmpty Dynamic -> Int -> IO ()
_expandNd (rets@(s:|_)) ops i = runManaged $ do
  st    <- managedState
  rets' <- mngNonEmpty rets
  ops'  <- mngNonEmpty ops
  liftIO $ Sig.c_expandNd st rets' ops' (fromIntegral i)
 where
  mngNonEmpty :: NonEmpty Dynamic -> Managed (Ptr (Ptr CTensor))
  mngNonEmpty = mapM toMPtr . NE.toList >=> mWithArray

  mWithArray :: [Ptr a] -> Managed (Ptr (Ptr a))
  mWithArray as = managed (FM.withArray as)

  toMPtr :: Dynamic -> Managed (Ptr CTensor)
  toMPtr d = managed (withForeignPtr (Sig.ctensor d))

instance IsList Dynamic where
  type Item Dynamic = HsReal
  toList = tensordata
  fromList l = newWithStorage1d (fromList l) 0 (genericLength l, 1)


-- | create a 1d Dynamic tensor from a list of elements.
--
-- FIXME construct this with TH, not by using 'setDim' inplace (one-by-one) which might be doing a second linear pass.
-- FIXME: CUDA doesn't like the storage allocation:
vectorEIO :: [HsReal] -> ExceptT String IO Dynamic
vectorEIO l = pure $ newWithStorage1d (fromList l) 0 (genericLength l, 1)

vectorE :: [HsReal] -> Either String Dynamic
vectorE = unsafePerformIO . runExceptT . vectorEIO
{-# NOINLINE vectorE #-}

vector :: [HsReal] -> Maybe Dynamic
vector = either (const Nothing) Just . vectorE

-- | create a 2d Dynamic tensor from a list of list of elements.
matrix :: [[HsReal]] -> ExceptT String IO Dynamic
matrix ls
  | null ls = lift (pure empty)
  | any ((ncols /=) . length) ls = ExceptT . pure $ Left "rows are not all the same length"
  | otherwise = pure $ newWithStorage2d (fromList l) 0 (nrows, ncols) (ncols, 1)
 where
  l = concat ls
  go vec (SomeDims ds) = resizeDim_ vec ds

  ncols :: Integral i => i
  ncols = genericLength (head ls)

  nrows :: Integral i => i
  nrows = genericLength ls

-- | create a 3d Dynamic tensor (ie: rectangular cuboid) from a nested list of elements.
{-# NOINLINE cuboid #-}
cuboid :: [[[HsReal]]] -> ExceptT String IO Dynamic
cuboid ls
  | isEmpty ls = lift (pure empty)
  | null ls || any null ls || any (any null) ls
                                   = ExceptT . pure . Left $ "can't accept empty lists"
  | innerDimCheck ncols        ls  = ExceptT . pure . Left $ "rows are not all the same length"
  | innerDimCheck ndepth (head ls) = ExceptT . pure . Left $ "columns are not all the same length"

  | otherwise = pure $ newWithStorage3d (fromList l) 0 (nrows, ncols * ndepth) (ncols, ndepth) (ndepth, 1)
 where
  l = concat (concat ls)
  go vec (SomeDims ds) = resizeDim_ vec ds >> pure vec

  isEmpty = \case
    []     -> True
    [[]]   -> True
    [[[]]] -> True
    _      -> False

  innerDimCheck :: Int -> [[x]] -> Bool
  innerDimCheck d = any ((/= d) . length)

  ndepth :: Integral i => i
  ndepth = genericLength (head (head ls))

  ncols :: Integral i => i
  ncols = genericLength (head ls)

  nrows :: Integral i => i
  nrows = genericLength ls


-- | create a 4d Dynamic tensor (ie: hyperrectangle) from a nested list of elements.
{-# NOINLINE hyper #-}
hyper :: [[[[HsReal]]]] -> ExceptT String IO Dynamic
hyper ls
  | isEmpty ls = lift (pure empty)
  | null ls
    || any null ls
    || any (any null) ls
    || any (any (any null)) ls           = ExceptT . pure . Left $ "can't accept empty lists"
  | innerDimCheck ntime (head (head ls)) = ExceptT . pure . Left $ "rows are not all the same length"
  | innerDimCheck ndepth      (head ls)  = ExceptT . pure . Left $ "cols are not all the same length"
  | innerDimCheck ncols             ls   = ExceptT . pure . Left $ "depths are not all the same length"

  | otherwise = pure $ newWithStorage4d (fromList l) 0
        (nrows, ncols * ndepth * ntime)
        (ncols, ndepth * ntime)
        (ndepth, ntime)
        (ntime, 1)
 where
  l = concat (concat (concat ls))
  go vec (SomeDims ds) = resizeDim_ vec ds >> pure vec

  isEmpty = \case
    []       -> True
    [[]]     -> True
    [[[]]]   -> True
    [[[[]]]] -> True
    _        -> False

  innerDimCheck :: Int -> [[x]] -> Bool
  innerDimCheck d = any ((/= d) . length)

  ntime :: Integral i => i
  ntime = genericLength (head (head (head ls)))

  ndepth :: Integral i => i
  ndepth = genericLength (head (head ls))

  ncols :: Integral i => i
  ncols = genericLength (head ls)

  nrows :: Integral i => i
  nrows = genericLength ls


