{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}

module Torch.Vision where

import qualified Codec.Picture as I
import Control.Exception.Safe
  ( SomeException (..),
    throwIO,
    try,
  )
import Control.Monad
  ( MonadPlus,
    forM_,
    when,
  )
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import Foreign.Marshal.Utils (copyBytes)
import Data.Int
import Data.Kind (Type)
import qualified Data.Vector.Storable as V
import Data.Word
import qualified Foreign.ForeignPtr as F
import qualified Foreign.Ptr as F
import GHC.Exts (IsList (fromList))
import qualified Language.C.Inline as C
import Pipes
import System.IO.Unsafe
import System.Random (mkStdGen, randoms)
import qualified Torch.DType as D
import Torch.Data.Pipeline
import Torch.Data.StreamedPipeline
import Torch.Functional hiding (take)
import qualified Torch.Functional as D
import Torch.Internal.Cast
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import Torch.NN
import Torch.Tensor
import qualified Torch.Tensor as D
import qualified Torch.TensorOptions as D
import qualified Torch.Typed.Vision as I
import Prelude hiding (max, min)
import qualified Prelude as P

C.include "<stdint.h>"

data MNIST (m :: Type -> Type) = MNIST
  { batchSize :: Int,
    mnistData :: I.MnistData
  }

instance Monad m => Datastream m Int (MNIST m) (Tensor, Tensor) where
  streamSamples MNIST {..} seed = Select $
    for (each [1 .. numIters]) $
      \iter -> do
        let from = (iter -1) * batchSize
            to = (iter * batchSize) - 1
            indexes = [from .. to]
            target = getLabels' batchSize mnistData indexes
        let input = getImages' batchSize 784 mnistData indexes
        yield (input, target)
    where
      numIters = I.length mnistData `Prelude.div` batchSize

instance Applicative m => Dataset m (MNIST m) Int (Tensor, Tensor) where
  getItem MNIST {..} ix =
    let indexes = [ix * batchSize .. (ix + 1) * batchSize - 1]
        imgs = getImages' batchSize 784 mnistData indexes
        labels = getLabels' batchSize mnistData indexes
     in pure (imgs, labels)

  keys MNIST {..} = fromList [0 .. I.length mnistData `Prelude.div` batchSize - 1]

getLabels' :: Int -> I.MnistData -> [Int] -> Tensor
getLabels' n mnist imageIdxs =
  asTensor $ map (I.getLabel mnist) . take n $ imageIdxs

getImages' ::
  Int -> -- number of observations in minibatch
  Int -> -- dimensionality of the data
  I.MnistData -> -- mnist data representation
  [Int] -> -- indices of the dataset
  Tensor
getImages' n dataDim mnist imageIdxs = unsafePerformIO $ do
  let (BSI.PS fptr off len) = I.images mnist
  t <-
    (cast2 LibTorch.empty_lo :: [Int] -> D.TensorOptions -> IO D.Tensor)
      [n, dataDim]
      (D.withDType D.UInt8 D.defaultOpts)
  D.withTensor t $ \ptr1 -> do
    F.withForeignPtr fptr $ \ptr2 -> do
      forM_ (zip [0 .. (n -1)] imageIdxs) $ \(i, idx) -> do
        copyBytes
          (F.plusPtr ptr1 (dataDim * i))
          (F.plusPtr ptr2 (off + 16 + dataDim * idx))
          dataDim
  return $ D.toType D.Float t

-- http://paulbourke.net/dataformats/asciiart/
grayScale10 = " .:-=+*#%@"

grayScale70 = reverse "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

-- Display an MNIST image tensor as ascii text
dispImage :: Tensor -> IO ()
dispImage img = do
  mapM
    ( \row ->
        mapM
          ( \col ->
              putChar $ grayScale !! (P.floor $ scaled !! row !! col)
          )
          [0, downSamp .. 27]
          >> putStrLn ""
    )
    [0, downSamp .. 27]
  pure ()
  where
    downSamp = 2
    grayScale = grayScale10
    paletteMax = (fromIntegral $ length grayScale) - 1.0
    img' = reshape [28, 28] img
    scaled :: [[Float]] =
      let (mn, mx) = (min img', max img')
       in asValue $ (img' - mn) / (mx - mn) * paletteMax

data PixelFormat
  = Y8
  | YF
  | YA8
  | RGB8
  | RGBF
  | RGBA8
  | YCbCr8
  | CMYK8
  | CMYK16
  | RGBA16
  | RGB16
  | Y16
  | YA16
  | Y32
  deriving (Show, Eq)

readImage :: FilePath -> IO (Either String (D.Tensor, PixelFormat))
readImage file =
  I.readImage file >>= \case
    Left err -> return $ Left err
    Right img' -> return $ Right $ (fromDynImage img', pixelFormat img')

readImageAsRGB8 :: FilePath -> IO (Either String D.Tensor)
readImageAsRGB8 file =
  I.readImage file >>= \case
    Left err -> return $ Left err
    Right img' -> return . Right . fromDynImage . I.ImageRGB8 . I.convertRGB8 $ img'

readImageAsRGB8WithScaling :: FilePath -> Int -> Int -> Bool -> IO (Either String (I.Image I.PixelRGB8, D.Tensor))
readImageAsRGB8WithScaling file width height keepAspectRatio =
  I.readImage file >>= \case
    Left err -> return $ Left err
    Right img' -> do
      let img = (resizeRGB8 width height keepAspectRatio) . I.convertRGB8 $ img'
      return $ Right (img, fromDynImage . I.ImageRGB8 $ img)

centerCrop :: Int -> Int -> I.Image I.PixelRGB8 -> I.Image I.PixelRGB8
centerCrop width height input = unsafePerformIO $ do
  let channel = 3 :: Int
      (I.Image org_w org_h org_vec) = input
      img@(I.Image w h vec) = I.generateImage (\_ _ -> (I.PixelRGB8 0 0 0)) width height :: I.Image I.PixelRGB8
      (org_fptr, org_len) = V.unsafeToForeignPtr0 org_vec
      org_whc = fromIntegral $ org_w * org_h * channel
      (fptr, len) = V.unsafeToForeignPtr0 vec
      whc = fromIntegral $ w * h * channel
  F.withForeignPtr org_fptr $ \ptr1 -> F.withForeignPtr fptr $ \ptr2 -> do
    let src = F.castPtr ptr1
        dst = F.castPtr ptr2
        iw = fromIntegral w
        ih = fromIntegral h
        iorg_w = fromIntegral org_w
        iorg_h = fromIntegral org_h
        ichannel = fromIntegral channel
    [C.block| void {
        uint8_t* src = $(uint8_t* src);
        uint8_t* dst = $(uint8_t* dst);
        int w = $(int iw);
        int h = $(int ih);
        int channel = $(int ichannel);
        int ow = $(int iorg_w);
        int oh = $(int iorg_h);
        int offsetx = (ow - w)/2;
        int offsety = (oh - h)/2;
        for(int y=0;y<h;y++){
          for(int x=0;x<w;x++){
            for(int c=0;c<channel;c++){
              int sy = y + offsety;
              int sx = x + offsetx;
              if(sx >= 0 && sx < ow &&
                 sy >= 0 && sy < oh){
                 dst[(y*w+x)*channel+c] = src[(sy*ow+sx)*channel+c];
              }
            }
          }
        }
    } |]
    return img

drawLine :: Int -> Int -> Int -> Int -> (Int, Int, Int) -> I.Image I.PixelRGB8 -> IO ()
drawLine x0 y0 x1 y1 (r, g, b) input = do
  let img@(I.Image w h vec) = input
      (fptr, len) = V.unsafeToForeignPtr0 vec
  F.withForeignPtr fptr $ \ptr2 -> do
    let iw = fromIntegral w
        ih = fromIntegral h
        ix0 = fromIntegral x0
        iy0 = fromIntegral y0
        ix1 = fromIntegral x1
        iy1 = fromIntegral y1
        ir = fromIntegral r
        ig = fromIntegral g
        ib = fromIntegral b
        dst = F.castPtr ptr2
    [C.block| void {
        uint8_t* dst = $(uint8_t* dst);
        int w = $(int iw);
        int h = $(int ih);
        int x0 = $(int ix0);
        int y0 = $(int iy0);
        int x1 = $(int ix1);
        int y1 = $(int iy1);
        int r = $(int ir);
        int g = $(int ig);
        int b = $(int ib);
        int channel = 3;
        int sign_x =  x1 - x0 >= 0 ? 1 : -1;
        int sign_y =  y1 - y0 >= 0 ? 1 : -1;
        int abs_x =  x1 - x0 >= 0 ? x1 - x0 : x0 - x1;
        int abs_y =  y1 - y0 >= 0 ? y1 - y0 : y0 - y1;
        if(abs_x>=abs_y){
          for(int x=x0;x!=x1;x+=sign_x){
            int y = (x-x0) * (y1-y0) / (x1-x0) + y0;
            if(y >=0 && y < h &&
               x >=0 && x < w) {
              dst[(y*w+x)*channel+0] = r;
              dst[(y*w+x)*channel+1] = g;
              dst[(y*w+x)*channel+2] = b;
            }
          }
        } else {
          for(int y=y0;y!=y1;y+=sign_y){
            int x = (y-y0) * (x1-x0) / (y1-y0) + x0;
            if(y >=0 && y < h &&
               x >=0 && x < w) {
              dst[(y*w+x)*channel+0] = r;
              dst[(y*w+x)*channel+1] = g;
              dst[(y*w+x)*channel+2] = b;
            }
          }
        }
    } |]

drawRect :: Int -> Int -> Int -> Int -> (Int, Int, Int) -> I.Image I.PixelRGB8 -> IO ()
drawRect x0 y0 x1 y1 (r, g, b) input = do
  drawLine x0 y0 (x1 + 1) y0 (r, g, b) input
  drawLine x0 y0 x0 (y1 + 1) (r, g, b) input
  drawLine x0 y1 (x1 + 1) y1 (r, g, b) input
  drawLine x1 y0 x1 (y1 + 1) (r, g, b) input

drawString :: String -> Int -> Int -> (Int, Int, Int) -> (Int, Int, Int) -> I.Image I.PixelRGB8 -> IO ()
drawString text x0 y0 (r, g, b) (br, bg, bb) input = do
  forM_ (zip [0 ..] text) $ \(i, ch) -> do
    drawChar (fromEnum ch) (x0 + i * 8) y0 (r, g, b) (br, bg, bb) input

drawChar :: Int -> Int -> Int -> (Int, Int, Int) -> (Int, Int, Int) -> I.Image I.PixelRGB8 -> IO ()
drawChar ascii_code x0 y0 (r, g, b) (br, bg, bb) input = do
  let img@(I.Image w h vec) = input
      (fptr, len) = V.unsafeToForeignPtr0 vec
  F.withForeignPtr fptr $ \ptr2 -> do
    let iw = fromIntegral w
        ih = fromIntegral h
        ix0 = fromIntegral x0
        iy0 = fromIntegral y0
        ir = fromIntegral r
        ig = fromIntegral g
        ib = fromIntegral b
        ibr = fromIntegral br
        ibg = fromIntegral bg
        ibb = fromIntegral bb
        dst = F.castPtr ptr2
        iascii_code = fromIntegral ascii_code
    [C.block| void {
        uint8_t* dst = $(uint8_t* dst);
        int w = $(int iw);
        int h = $(int ih);
        int x0 = $(int ix0);
        int y0 = $(int iy0);
        int r = $(int ir);
        int g = $(int ig);
        int b = $(int ib);
        int br = $(int ibr);
        int bg = $(int ibg);
        int bb = $(int ibb);
        int ascii_code = $(int iascii_code);
        int channel = 3;
        int char_width = 8;
        int char_height = 8;
        char fonts[95][8] = { // 0x20 to 0x7e
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
            { 0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00},
            { 0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
            { 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00},
            { 0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00},
            { 0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00},
            { 0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00},
            { 0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00},
            { 0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00},
            { 0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00},
            { 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00},
            { 0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00},
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06},
            { 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00},
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00},
            { 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00},
            { 0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00},
            { 0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00},
            { 0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00},
            { 0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00},
            { 0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00},
            { 0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00},
            { 0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00},
            { 0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00},
            { 0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00},
            { 0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00},
            { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00},
            { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06},
            { 0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00},
            { 0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00},
            { 0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00},
            { 0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00},
            { 0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00},
            { 0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00},
            { 0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00},
            { 0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00},
            { 0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00},
            { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00},
            { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00},
            { 0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00},
            { 0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00},
            { 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},
            { 0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00},
            { 0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00},
            { 0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00},
            { 0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00},
            { 0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00},
            { 0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00},
            { 0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00},
            { 0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00},
            { 0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00},
            { 0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00},
            { 0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},
            { 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00},
            { 0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00},
            { 0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00},
            { 0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00},
            { 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00},
            { 0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00},
            { 0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00},
            { 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00},
            { 0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00},
            { 0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00},
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF},
            { 0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00},
            { 0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00},
            { 0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00},
            { 0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00},
            { 0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6E, 0x00},
            { 0x00, 0x00, 0x1E, 0x33, 0x3f, 0x03, 0x1E, 0x00},
            { 0x1C, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0F, 0x00},
            { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F},
            { 0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00},
            { 0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},
            { 0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E},
            { 0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00},
            { 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},
            { 0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00},
            { 0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00},
            { 0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00},
            { 0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F},
            { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78},
            { 0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00},
            { 0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00},
            { 0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00},
            { 0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00},
            { 0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00},
            { 0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00},
            { 0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00},
            { 0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F},
            { 0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00},
            { 0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00},
            { 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00},
            { 0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00},
            { 0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00} 
          };
        for(int y=y0;y<y0+char_height;y++){
          for(int x=x0;x<x0+char_width;x++){
            if(y >=0 && y < h &&
               x >=0 && x < w) {
              int dx = x-x0;
              int dy = y-y0;
              int bit = 
                ascii_code > 0x20 && ascii_code < 0x7f ?
                fonts[ascii_code-0x20][dy] & (0x1 << dx) :
                0;
              if (bit) {
                dst[(y*w+x)*channel+0] = r;
                dst[(y*w+x)*channel+1] = g;
                dst[(y*w+x)*channel+2] = b;
              } else {
                dst[(y*w+x)*channel+0] = br;
                dst[(y*w+x)*channel+1] = bg;
                dst[(y*w+x)*channel+2] = bb;
              }
            }
          }
        }
    } |]

resizeRGB8 :: Int -> Int -> Bool -> I.Image I.PixelRGB8 -> I.Image I.PixelRGB8
resizeRGB8 width height keepAspectRatio input = unsafePerformIO $ do
  let channel = 3 :: Int
      (I.Image org_w org_h org_vec) = input
      img@(I.Image w h vec) = I.generateImage (\_ _ -> (I.PixelRGB8 0 0 0)) width height :: I.Image I.PixelRGB8
      (org_fptr, org_len) = V.unsafeToForeignPtr0 org_vec
      org_whc = fromIntegral $ org_w * org_h * channel
      (fptr, len) = V.unsafeToForeignPtr0 vec
      whc = fromIntegral $ w * h * channel
  F.withForeignPtr org_fptr $ \ptr1 -> F.withForeignPtr fptr $ \ptr2 -> do
    let src = F.castPtr ptr1
        dst = F.castPtr ptr2
        iw = fromIntegral w
        ih = fromIntegral h
        iorg_w = fromIntegral org_w
        iorg_h = fromIntegral org_h
        ichannel = fromIntegral channel
        ckeepAspectRatio = if keepAspectRatio then 1 else 0
    [C.block| void {
        uint8_t* src = $(uint8_t* src);
        uint8_t* dst = $(uint8_t* dst);
        int w = $(int iw);
        int h = $(int ih);
        int channel = $(int ichannel);
        int ow = $(int iorg_w);
        int oh = $(int iorg_h);
        int keepAspectRatio = $(int ckeepAspectRatio);
        if(keepAspectRatio){
          int t0h = h;
          int t0w = ow * h / oh;
          int t1h = oh * w / ow;
          int t1w = w;
          if (t0w > w) {
            int offset = (h - (oh * w / ow))/2;
            for(int y=offset;y<h-offset;y++){
              for(int x=0;x<w;x++){
                for(int c=0;c<channel;c++){
                  int sy = (y-offset) * ow / w;
                  int sx = x * ow / w;
                  if(sy >= 0 && sy < oh){
                    dst[(y*w+x)*channel+c] = src[(sy*ow+sx)*channel+c];
                  }
                }
              }
            }
          } else {
            int offset = (w - (ow * h / oh))/2;
            for(int y=0;y<h;y++){
              for(int x=offset;x<w-offset;x++){
                for(int c=0;c<channel;c++){
                  int sy = y * oh / h;
                  int sx = (x-offset) * oh / h;
                  if(sx >= 0 && sx < ow){
                    dst[(y*w+x)*channel+c] = src[(sy*ow+sx)*channel+c];
                  }
                }
              }
            }
          }
        } else {
          for(int y=0;y<h;y++){
            for(int x=0;x<w;x++){
              for(int c=0;c<channel;c++){
                int sy = y * oh / h;
                int sx = x * ow / w;
                dst[(y*w+x)*channel+c] = src[(sy*ow+sx)*channel+c];
              }
            }
          }
        }
    } |]
    return img

pixelFormat :: I.DynamicImage -> PixelFormat
pixelFormat image = case image of
  I.ImageY8 _ -> Y8
  I.ImageYF _ -> YF
  I.ImageYA8 _ -> YA8
  I.ImageRGB8 _ -> RGB8
  I.ImageRGBF _ -> RGBF
  I.ImageRGBA8 _ -> RGBA8
  I.ImageYCbCr8 _ -> YCbCr8
  I.ImageCMYK8 _ -> CMYK8
  I.ImageCMYK16 _ -> CMYK16
  I.ImageRGBA16 _ -> RGBA16
  I.ImageRGB16 _ -> RGB16
  I.ImageY16 _ -> Y16
  I.ImageYA16 _ -> YA16
  I.ImageY32 _ -> Y32

fromDynImage :: I.DynamicImage -> D.Tensor
fromDynImage image = unsafePerformIO $ case image of
  I.ImageY8 (I.Image width height vec) -> createTensor width height 1 D.UInt8 1 vec
  I.ImageYF (I.Image width height vec) -> createTensor width height 1 D.Float 4 vec
  I.ImageYA8 (I.Image width height vec) -> createTensor width height 2 D.UInt8 1 vec
  I.ImageRGB8 (I.Image width height vec) -> createTensor width height 3 D.UInt8 1 vec
  I.ImageRGBF (I.Image width height vec) -> createTensor width height 3 D.Float 4 vec
  I.ImageRGBA8 (I.Image width height vec) -> createTensor width height 4 D.UInt8 1 vec
  I.ImageYCbCr8 (I.Image width height vec) -> createTensor width height 3 D.UInt8 1 vec
  I.ImageCMYK8 (I.Image width height vec) -> createTensor width height 4 D.UInt8 1 vec
  I.ImageCMYK16 (I.Image width height vec) -> createTensorU16to32 width height 4 D.Int32 vec
  I.ImageRGBA16 (I.Image width height vec) -> createTensorU16to32 width height 4 D.Int32 vec
  I.ImageRGB16 (I.Image width height vec) -> createTensorU16to32 width height 3 D.Int32 vec
  I.ImageY16 (I.Image width height vec) -> createTensorU16to32 width height 1 D.Int32 vec
  I.ImageYA16 (I.Image width height vec) -> createTensorU16to32 width height 2 D.Int32 vec
  I.ImageY32 (I.Image width height vec) -> createTensorU32to64 width height 1 D.Int64 vec
  where
    createTensor width height channel dtype dtype_size vec = do
      t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor) [1, height, width, channel] $ D.withDType dtype D.defaultOpts
      D.withTensor t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = width * height * channel * dtype_size
        F.withForeignPtr fptr $ \ptr2 -> do
          copyBytes (F.castPtr ptr1) (F.castPtr ptr2) whc
          return t
    createTensorU16to32 width height channel dtype vec = do
      t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor) [1, height, width, channel] $ D.withDType dtype D.defaultOpts
      D.withTensor t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = fromIntegral $ width * height * channel
        F.withForeignPtr fptr $ \ptr2 -> do
          let src = F.castPtr ptr2
              dst = F.castPtr ptr1
          [C.block| void {
              uint16_t* src = $(uint16_t* src);
              int32_t* dst = $(int32_t* dst);
              for(int i=0;i<$(int whc);i++){
                 dst[i] = src[i];
              }
          } |]
          return t
    createTensorU32to64 width height channel dtype vec = do
      t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor) [1, height, width, channel] $ D.withDType dtype D.defaultOpts
      D.withTensor t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = fromIntegral $ width * height * channel
        F.withForeignPtr fptr $ \ptr2 -> do
          let src = F.castPtr ptr2
              dst = F.castPtr ptr1
          [C.block| void {
              uint32_t* src = $(uint32_t* src);
              int64_t* dst = $(int64_t* dst);
              for(int i=0;i<$(int whc);i++){
                 dst[i] = src[i];
              }
          } |]
          return t

fromImages :: [I.Image I.PixelRGB8] -> IO D.Tensor
fromImages imgs = do
  let num_imgs = length imgs
      channel = 3
      (I.Image width height _) = head imgs
  when (num_imgs == 0) $ do
    throwIO $ userError "The number of images should be greater than 0."
  t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor) [num_imgs, height, width, channel] $ D.withDType D.UInt8 D.defaultOpts
  D.withTensor t $ \ptr1 -> do
    forM_ (zip [0 ..] imgs) $ \(idx, (I.Image width' height' vec)) -> do
      let (fptr, len) = V.unsafeToForeignPtr0 vec
          whc = width * height * channel
      when (len /= whc) $ do
        throwIO $ userError "vector's length is not the same as tensor' one."
      when (width /= width') $ do
        throwIO $ userError "image's width is not the same as first image's one"
      when (height /= height') $ do
        throwIO $ userError "image's height is not the same as first image's one"
      F.withForeignPtr fptr $ \ptr2 -> do
        copyBytes (F.plusPtr (F.castPtr ptr1) (whc * idx)) ptr2 len
  return t

writeImage :: forall p. I.Pixel p => Int -> Int -> Int -> p -> D.Tensor -> IO (I.Image p)
writeImage width height channel pixel tensor = do
  let img@(I.Image w h vec) = I.generateImage (\_ _ -> pixel) width height :: I.Image p
  D.withTensor tensor $ \ptr1 -> do
    let (fptr, len) = V.unsafeToForeignPtr0 vec
        whc = width * height * channel
    if (len /= whc)
      then throwIO $ userError $ "vector's length(" ++ show len ++ ") is not the same as tensor' one."
      else do
        F.withForeignPtr fptr $ \ptr2 -> do
          copyBytes (F.castPtr ptr2) (F.castPtr ptr1) len
          return img

writeBitmap :: FilePath -> D.Tensor -> IO ()
writeBitmap file tensor = do
  case (D.shape tensor, D.dtype tensor) of
    ([1, height, width, 1], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([1, height, width], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([1, height, width, 3], D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writeBitmap file
    ([height, width, 1], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([height, width], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([height, width, 3], D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writeBitmap file
    format -> throwIO $ userError $ "Can not write a image. " ++ show format ++ " is unsupported format."

writePng :: FilePath -> D.Tensor -> IO ()
writePng file tensor = do
  case (D.shape tensor, D.dtype tensor) of
    ([1, height, width, 1], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([1, height, width], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([1, height, width, 3], D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writePng file
    ([height, width, 1], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([height, width], D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([height, width, 3], D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writePng file
    format -> throwIO $ userError $ "Can not write a image. " ++ show format ++ " is unsupported format."

-- [batch, height, width, channel] -> [batch, channel, height, width]
hwc2chw :: D.Tensor -> D.Tensor
hwc2chw = D.permute [0, 3, 1, 2]

-- [batch, channel, height, width] -> [batch, height, width, channel]
chw2hwc :: D.Tensor -> D.Tensor
chw2hwc = D.permute [0, 2, 3, 1]

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123
