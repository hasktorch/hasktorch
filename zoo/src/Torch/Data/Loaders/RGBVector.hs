{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Data.Loaders.RGBVector
  ( Normalize(..)
  , file2rgb
  , rgb2list
  , assertList
  ) where

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

import Torch.Data.Loaders.Logging

type HsReal = Double
type MRGBVector s = MVector s (MVector s (MVector s HsReal))
type RGBVector = Vector (Vector (Vector HsReal))

modulename = "Torch.Data.Loaders.RGBVector"

newtype Normalize = Normalize Bool
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | load an RGB PNG image into a Torch tensor
rgb2list
  :: forall h w . (All KnownDim '[h, w], All KnownNat '[h, w])
  => Proxy '(h, w)
  -> Normalize
  -> FilePath
  -> ExceptT String IO [[[HsReal]]]
rgb2list hwp (Normalize donorm) fp = do
  pxs <- file2rgb hwp fp
  lift $ assertPixels pxs
  ExceptT $ do
    vec <- mkRGBVec
    -- threadDelay 1000
    fillFrom pxs $ \chw px -> do
      let pxfin = prep px
      writePx vec chw pxfin

    lst <- freezeList vec
    assertList modulename (concat (concat lst))
    pure $ Right lst
 where
  prep w =
    if donorm
    then w / 255
    else w

  (height, width) = reifyHW hwp

  mkRGBVec :: PrimMonad m => m (MRGBVector (PrimState m))
  mkRGBVec = M.replicateM 3 (M.replicateM height (M.unsafeNew width))

  writePx
    :: PrimMonad m
    => MRGBVector (PrimState m)
    -> (Int, Int, Int)
    -> HsReal
    -> m ()
  writePx channels (c, h, w) px
    = M.unsafeRead channels c
    >>= \rows -> M.unsafeRead rows h
    >>= \cols -> M.unsafeWrite cols w px

  readPx
    :: PrimMonad m
    => MRGBVector (PrimState m)
    -> (Int, Int, Int)
    -> m HsReal
  readPx channels (c, h, w)
    = M.unsafeRead channels c
    >>= \rows -> M.unsafeRead rows h
    >>= \cols -> M.unsafeRead cols w

  freezeList
    :: PrimMonad m => MRGBVector (PrimState m) -> m [[[HsReal]]]
  freezeList mvecs = do
    readN mvecs 3 $ \mframe ->
      readN mframe height $ \mrow ->
        readN mrow width pure



readNfreeze :: PrimMonad m => MVector (PrimState m) a -> Int -> (a -> m b) -> m (Vector b)
readNfreeze mvec n op =
  V.fromListN n <$> readN mvec n op

readN :: PrimMonad m => MVector (PrimState m) a -> Int -> (a -> m b) -> m [b]
readN mvec n op = mapM (M.read mvec >=> op) [0..n-1]



fillFrom :: (Num y, PrimMonad m) => [((Int, Int), (Int, Int, Int))] -> ((Int, Int, Int) -> y -> m ()) -> m ()
fillFrom pxs filler =
  forM_ pxs $ \((h, w), (r, g, b)) ->
    forM_ (zip [0..] [r,g,b]) $ \(c, px) ->
      filler (c, h, w) (fromIntegral px)

file2rgb
  :: forall h w hw rgb
  . (All KnownDim '[h, w], All KnownNat '[h, w])
  => hw ~ (Int, Int)
  => rgb ~ (Int, Int, Int)
  => Proxy '(h, w)
  -> FilePath
  -> ExceptT String IO [(hw, rgb)]
file2rgb hwp fp = do
#ifdef USE_GD
  im <- lift $ GD.loadPngFile fp
  forM [(h, w) | h <- [0.. height - 1], w <- [0.. width - 1]] $ \(h, w) -> do
    (r,g,b,_) <- lift $ GD.toRGBA <$> GD.getPixel (h,w) im
#else
  im <- JP.convertRGB8 <$> ExceptT (JP.readPng fp)
  forM [(h, w) | h <- [0.. height - 1], w <- [0.. width - 1]] $ \(h, w) -> do
    let JP.PixelRGB8 r g b = JP.pixelAt im h w
#endif
    -- lift $ print (r, g, b)
    pure ((h, w), (fromIntegral r, fromIntegral g, fromIntegral b))
 where
  (height, width) = reifyHW hwp

assertPixels :: [((Int, Int), (Int, Int, Int))] -> IO ()
assertPixels pxs = do
  if all ((\(r, g, b) -> all (==0) [r, g, b]). snd) pxs
  then throwString $ mkError modulename "IMAGE ALL ZEROS!"
  else
    if all ((\(r, g, b) -> any (\x -> x < 0 || x > 255) [r, g, b]). snd) pxs
    then throwString $ mkError modulename "IMAGE OUT OF PIXEL BOUNDS!"
    else pure ()

assertList :: String -> [HsReal] -> IO ()
assertList hdr rs = do
  let
    oob = filter (\x -> x < -0.1 || x > 255.1) rs
  if not (null oob)
  then throwString $ show ({-oob,-} length oob, length rs, mkError hdr "OOB found!")
  else
    if all (== 0) rs
    then throwString $ mkError hdr "all-zeros found!"
    else pure () -- $ print $ mkInfo hdr "reified vals good"


reifyHW
  :: forall h w
  . (All KnownDim '[h, w], All KnownNat '[h, w])
  => Proxy '(h, w)
  -> (Int, Int)
reifyHW _ = (fromIntegral (dimVal (dim :: Dim h)), fromIntegral (dimVal (dim :: Dim w)))


