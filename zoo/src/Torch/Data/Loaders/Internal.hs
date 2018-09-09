{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TupleSections #-}
module Torch.Data.Loaders.Internal where

import GHC.Int
import Data.Proxy
import Data.Vector (Vector)
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
import qualified Torch.Double.Dynamic as Dynamic
#endif

import Torch.Data.Loaders.RGBVector
import Data.List

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
rgb2torch norm fp = ExceptT . pure . cuboid =<< rgb2list (Proxy :: Proxy '(h, w)) norm fp


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

