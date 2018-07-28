{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE CPP #-}
module Torch.Data.Loaders.Cifar10 where

import Control.Monad
import Data.Char
import Data.List
import System.Directory
import System.FilePath
import Control.Monad.Trans.Except
import Control.Monad.Trans.Class
import ListT
import Data.Singletons
import Numeric.Dimensions
import Data.Singletons.Prelude.Enum
import GHC.TypeLits

import qualified Data.Vector as V
import qualified Codec.Picture as JP
import qualified Graphics.GD as GD

#ifdef CUDA
import Torch.Cuda.Double
import qualified Torch.Cuda.Long as Long
import Control.Concurrent (threadDelay)
#else
import Torch.Double
import qualified Torch.Long as Long
#endif

-- This should be replaced with a download-aware cache.
default_cifar_path :: FilePath
default_cifar_path = "/mnt/lake/datasets/cifar-10"

data Mode = Test | Train
  deriving (Eq, Enum, Ord, Show, Bounded)

testLength  :: Proxy 'Test -> Proxy 1000
testLength _ = Proxy

trainLength :: Proxy 'Train -> Proxy 5000
trainLength _ = Proxy

data Category
  = Airplane    -- 0
  | Automobile  -- 2
  | Bird        -- 3
  | Cat         -- 4
  | Deer        -- 5
  | Dog         -- 6
  | Frog        -- 7
  | Horse       -- 8
  | Ship        -- 9
  | Truck       -- 10
  deriving (Eq, Enum, Ord, Show, Bounded)

category_path :: FilePath -> Mode -> Category -> FilePath
category_path cifarpath m c = intercalate "/"
  [ cifarpath
  , toLower <$> show m
  , toLower <$> show c
  ]

im2torch :: FilePath -> ExceptT String IO (Tensor '[3, 32, 32])
im2torch fp = do
  -- !im <- JP.convertRGB8 <$> ExceptT (JP.readPng fp)
  !im <- lift $ GD.loadPngFile fp
  !t <- lift new
  lift $ forM_ [(h, w) | h <- [0..31], w <- [0..31]] $ \(h, w) -> do
    -- let JP.PixelRGB8 r g b = JP.pixelAt im h w
    (r,g,b,_) <- GD.toRGBA <$> GD.getPixel (h,w) im
    forM_ (zip [0..] [r, g, b]) $ \(c, px) ->
      setDim'_ t (someDimsVal $ fromIntegral <$> [c, h, w]) (fromIntegral px)
  pure t

cat2torch :: FilePath -> ListT IO (Tensor '[3, 32, 32])
cat2torch fp = do
  ims <- lift $ filter ((== ".png") . takeExtension) <$> getDirectoryContents fp
  im <- fromFoldable (Data.List.take 20 ims)

  lift (runExceptT (im2torch (fp </> im))) >>= \case
    Left _ -> mempty
    Right t -> pure t

onehot
  :: forall c sz
  . (Ord c, Bounded c, Enum c) -- , sz ~ FromEnum (MaxBound c), KnownDim sz, KnownNat sz)
  => c
  -> LongTensor '[10] -- '[FromEnum (MaxBound c)]
onehot c
  = Long.unsafeVector
  $ V.toList
  $ onehotv c

onehotv :: forall i c . (Integral i, Ord c, Bounded c, Enum c) => c -> V.Vector i
onehotv c =
  V.generate
    (fromEnum (maxBound :: c) + 1)
    (fromIntegral . fromEnum . (== fromEnum c))


cifar10set :: FilePath -> Mode -> ListT IO (Tensor '[3, 32, 32], Category)
cifar10set fp m = do
  c <- fromFoldable [minBound..maxBound :: Category]
  t <- cat2torch (category_path fp m c)
  pure (t, c)

defaultCifar10set :: Mode -> ListT IO (Tensor '[3, 32, 32], Category)
defaultCifar10set = cifar10set default_cifar_path


#ifdef DEBUG
-- example use-case
main :: IO ()
main = do
  forM_ [minBound..maxBound :: Mode] $ \m -> do
    l <- toList (defaultCifar10set m)
    print $ length l
#endif

