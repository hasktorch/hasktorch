{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
module Torch.Data.Loaders.Internal where

import Data.Vector (Vector)
import Control.Concurrent (threadDelay)
import Control.Monad (forM_, filterM)
import Control.Monad.Trans.Class
import Control.Monad.Trans.Except
import GHC.Conc (getNumProcessors)
import Numeric.Dimensions
import System.Random.MWC (GenIO)
import System.Random.MWC.Distributions (uniformShuffle)
import System.Directory (listDirectory, doesDirectoryExist)
import System.FilePath ((</>), takeExtension)

import qualified Data.Vector as V

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
#else
import Torch.Double
import qualified Torch.Long as Long
#endif


-- | asyncronously map across a pool with a maximum level of concurrency
mapPool :: Traversable t => Int -> (a -> IO b) -> t a -> IO (t b)
mapPool mx fn xs = do
  sem <- MSem.new mx
  Async.mapConcurrently (MSem.with sem . fn) xs

-- | load an RGB PNG image into a Torch tensor
rgb2torch
  :: forall h w . (KnownDim h, KnownDim w)
  => FilePath -> ExceptT String IO (Tensor '[3, h, w])
rgb2torch fp = do
  t <- lift new
#ifdef USE_GD
  im <- lift $ GD.loadPngFile fp
  setPixels $ \(h, w) -> do
    (r,g,b,_) <- GD.toRGBA <$> GD.getPixel (h,w) im
#else
  im <- JP.convertRGB8 <$> ExceptT (JP.readPng fp)
  setPixels $ \(h, w) -> do
    let JP.PixelRGB8 r g b = JP.pixelAt im h w
#endif
    forM_ (zip [0..] [r, g, b]) $ \(c, px) ->
      setDim'_ t (someDimsVal $ fromIntegral <$> [c, h, w]) (fromIntegral px)
  pure t
 where
  setPixels :: ((Int, Int) -> IO ()) -> ExceptT String IO ()
  setPixels
    = lift . forM_ [(h, w) | h <- [0.. fromIntegral (dimVal (dim :: Dim h)) - 1]
                           , w <- [0.. fromIntegral (dimVal (dim :: Dim w)) - 1]]

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

#ifdef DEBUG
main = do
  nprocs <- getNumProcessors
  putStrLn $ "number of cores: " ++ show nprocs
  mapPool nprocs (\x -> threadDelay 100000 >> print x >> pure x) [1..100]
#endif
