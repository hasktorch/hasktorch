{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TupleSections #-}
module Torch.Data.Loaders.Internal where

import Prelude hiding (print, putStrLn)
import qualified Prelude as P (print, putStrLn)
import GHC.Int
import Data.Proxy
import Data.Vector (Vector)
import qualified Data.List as List ((!!))
import Control.Concurrent (threadDelay)
import Control.Monad -- (forM_, filterM)
import Control.Monad.Trans.Class
import Control.Monad.Trans.Except
import Control.Exception.Safe
import Control.DeepSeq
import GHC.Conc (getNumProcessors)
import GHC.TypeLits (KnownNat)
import Numeric.Dimensions
import System.Random.MWC (GenIO)
import System.Random.MWC.Distributions (uniformShuffle)
import System.Directory (listDirectory, doesDirectoryExist)
import System.FilePath ((</>), takeExtension)
import Control.Concurrent

import Control.Monad.Primitive
import qualified Data.Vector as V
import Data.Vector.Mutable (MVector)
import qualified Data.Vector.Mutable as M

import qualified Control.Concurrent.MSem as MSem (new, with)
import qualified Control.Concurrent.Async as Async

#ifdef USE_GD
import qualified Graphics.GD as GD
#else
import qualified Codec.Picture as JP
#endif

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
import qualified Torch.Cuda.Double.Dynamic as Dynamic
import qualified Torch.Double.Dynamic as CPU
#else
import Torch.Double
import qualified Torch.Long as Long
import qualified Torch.Double.Storage as Storage
import qualified Torch.Double.Dynamic as Dynamic
#endif

import Torch.Data.Loaders.RGBVector
import Data.List
import Text.Printf

-- | asyncronously map across a pool with a maximum level of concurrency
mapPool :: Traversable t => Int -> (a -> IO b) -> t a -> IO (t b)
mapPool mx fn xs = do
  sem <- MSem.new mx
  Async.mapConcurrently (MSem.with sem . fn) xs

-- | load an RGB PNG image into a Torch tensor
rgb2torch
  :: forall h w . (All KnownDim '[h, w], All KnownNat '[h, w])
  => Normalize
  -> FilePath
  -> ExceptT String IO (Tensor '[3, h, w])
rgb2torch norm fp = do
  x <- rgb2list (Proxy :: Proxy '(h, w)) norm fp
  -- lift $ do
  --   forM_ [0..2] $ \ch -> do
  --     forM_ [0..31] $ \r -> do
  --       printf "\n>> "
  --       forM_ [0..31] $ \c -> do
  --         let v = (x List.!! ch) List.!! r List.!! c
  --         printf ((if v < 0 then " " else "  ")++"%.4f") v
  --     putStrLn ("\n" :: String)

  -- y <- lift $ do
  --   print "???"
  --   print x
  --   y <- unsafeCuboid x
  --   -- y <- asStatic <$> ioCuboid x
  --   print y
  --   print "should have printed"

  --   throwString "NO!"
  --   pure y
  -- ExceptT . pure . Right $ y
  lift $ unsafeCuboid x
 where
  -- | create a 3d Dynamic tensor (ie: rectangular cuboid) from a nested list of elements.
  ioCuboid :: [[[HsReal]]] -> IO Dynamic
  ioCuboid ls = do
    let l = concat (concat ls)
    vec <- vector (deepseq l l)
    v <- go vec (someDimsVal [nrows, ncols, ndepth])

    -- st <- Storage.fromList (deepseq l l)
    -- Storage.tensordata st >>= print

    -- VALID, BUT WRONG
    -- Dynamic.newWithStorage3d st 0 (nrows, ncols*ndepth) (ncols, ndepth) (ndepth, 1)
    -- Dynamic.newWithStorage3d st 0 (nrows, 1) (ncols, nrows) (ndepth, nrows*ncols)
    --
    -- INVALID
    -- Dynamic.newWithStorage3d st 0 (ndepth, ncols*ndepth) (ncols, ndepth) (nrows, 1)
    pure v
   where
    go vec (SomeDims ds) = Dynamic._resizeDim vec ds >> pure vec

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

  vector :: [HsReal] -> IO Dynamic
  vector l = do
    -- res <- Dynamic.new' (someDimsVal [genericLength l])
    -- mapM_  (upd res) (zip [0..genericLength l - 1] l)
    -- pure res
    -- -- IMPORTANT: This is safe for CPU. For GPU, I think we need to allocate and marshal a haskell list into cuda memory before continuing.
    st <- Storage.fromList (deepseq l l)
    Dynamic.newWithStorage1d st 0 (genericLength l, 1)
   where
    upd :: Dynamic -> (Word, HsReal) -> IO ()
    upd t (idx, v) =
      let ix = [idx]
      in Dynamic.setDim'_ t (someDimsVal (deepseq ix ix)) v


-- | Given a folder with subfolders of category images, return a uniform-randomly
-- shuffled list of absolute filepaths with the corresponding category.
shuffleCatFolders
  :: forall c
  .  GenIO                        -- ^ generator for shuffle
  -> (FilePath -> Maybe c)        -- ^ how to convert a subfolder into a category
  -> FilePath                     -- ^ absolute path of the dataset
  -> IO (Vector (c, FilePath))    -- ^ shuffled list
shuffleCatFolders g cast path = do
  cats <- filterM (doesDirectoryExist . (path </>)) =<< listDirectory path
  imgfiles <- sequence $ catContents <$> cats
  uniformShuffle (V.concat imgfiles) g
 where
  catContents :: FilePath -> IO (Vector (c, FilePath))
  catContents catFP =
    case cast catFP of
      Nothing -> pure mempty
      Just c ->
        let
          fdr = path </> catFP
          asPair img = (c, fdr </> img)
        in
          V.fromList . fmap asPair . filter isImage
          <$> listDirectory fdr

-- | verifies that an absolute filepath is an image
isImage :: FilePath -> Bool
isImage = (== ".png") . takeExtension

