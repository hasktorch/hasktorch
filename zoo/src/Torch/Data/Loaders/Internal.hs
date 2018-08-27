{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Data.Loaders.Internal where

import Control.Concurrent (threadDelay)
import Control.Monad (forM_)
import Control.Monad.Trans.Class
import Control.Monad.Trans.Except
import GHC.Conc (getNumProcessors)
import Numeric.Dimensions

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


mapPool :: Traversable t => Int -> (a -> IO b) -> t a -> IO (t b)
mapPool mx fn xs = do
  sem <- MSem.new mx
  Async.mapConcurrently (MSem.with sem . fn) xs

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

main = do
  nprocs <- getNumProcessors
  putStrLn $ "number of cores: " ++ show nprocs
  mapPool nprocs (\x -> threadDelay 100000 >> print x >> pure x) [1..100]
