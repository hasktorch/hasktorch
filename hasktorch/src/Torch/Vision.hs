{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}

module Torch.Vision where

import Prelude hiding (min, max)
import qualified Prelude as P


import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                , throwIO
                                                )
import           Control.Monad                  ( MonadPlus 
                                                , forM_
                                                , when
                                                )
import           Torch.Internal.Cast
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import qualified Torch.DType as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorOptions           as D
import qualified Torch.Functional              as D
import qualified Foreign.ForeignPtr            as F
import qualified Foreign.Ptr                   as F
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Language.C.Inline as C
import           System.IO.Unsafe
import           Data.Int
import           Data.Word
import qualified Codec.Picture                 as I
import qualified Data.Vector.Storable          as V
import           System.Random (mkStdGen, randoms)

import Pipes

import qualified Torch.Typed.Vision as I
import Torch.Functional hiding (take)
import Torch.Tensor
import Torch.NN

import Torch.Data.StreamedPipeline 
import Torch.Data.Pipeline
import Pipes.Prelude (repeatM)
import GHC.Exts (IsList(fromList))

C.include "<stdint.h>"

data Mnist = Mnist { batchSize :: Int
                   , mnistData :: I.MnistData
                   }

instance (MonadPlus m, MonadBase IO m) => Datastream m Int Mnist (Tensor, Tensor) where
  streamBatch Mnist{..} seed = Select $ 
    for (each [1..numIters]) $ \iter -> do 
      let from = (iter-1) * batchSize
          to = (iter * batchSize) - 1
          indexes = [from .. to]
          target = getLabels' batchSize  mnistData indexes
      let input = getImages' batchSize 784 mnistData indexes
      yield (input, target)

      where numIters = I.length mnistData `Prelude.div` batchSize
            
instance Applicative m => Dataset m Mnist Int (Tensor, Tensor) where
  getItem Mnist{..} ix =  
    let
      indexes = [ix * batchSize .. (ix+1) * batchSize - 1]
      imgs = getImages' batchSize 784 mnistData indexes
      labels = getLabels' batchSize mnistData indexes
    in pure (imgs, labels)
  -- size Mnist{..} = I.length mnistData `Prelude.div` batchSize
  keys Mnist{..} = fromList [ 0 .. I.length mnistData `Prelude.div` batchSize - 1 ]


getLabels' :: Int -> I.MnistData -> [Int] -> Tensor
getLabels' n mnist imageIdxs =
  asTensor $ map (I.getLabel mnist) . take n $ imageIdxs

getImages' ::
  Int -- number of observations in minibatch
  -> Int -- dimensionality of the data
  -> I.MnistData -- mnist data representation
  -> [Int] -- indices of the dataset
  -> Tensor
getImages' n dataDim mnist imageIdxs = unsafePerformIO $ do
  let (BSI.PS fptr off len) = I.images mnist
  t <- (cast2 LibTorch.empty_lo :: [Int] -> D.TensorOptions -> IO D.Tensor)
         [n, dataDim]
         (D.withDType D.UInt8 D.defaultOpts)
  D.withTensor t $ \ptr1 -> do
    F.withForeignPtr fptr $ \ptr2 -> do
      forM_ (zip [0..(n-1)] imageIdxs) $ \(i,idx) -> do
        BSI.memcpy
          (F.plusPtr ptr1 (dataDim*i))
          (F.plusPtr ptr2 (off+16+dataDim*idx))
          dataDim
  return $ D.toType D.Float t

-- http://paulbourke.net/dataformats/asciiart/
grayScale10 = " .:-=+*#%@"
grayScale70 = reverse "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

-- Display an MNIST image tensor as ascii text
dispImage :: Tensor -> IO ()
dispImage img = do
    mapM (\row ->
        mapM (\col -> 
            putChar $ grayScale !! (P.floor $ scaled !! row !! col)
            ) [0,downSamp..27] >>  putStrLn ""
        ) [0,downSamp..27]
    pure ()
    where 
        downSamp = 2
        grayScale = grayScale10
        paletteMax = (fromIntegral $ length grayScale) - 1.0
        img' = reshape [28, 28] img
        scaled :: [[Float]] = let (mn, mx) = (min img', max img')  
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

readImage :: FilePath -> IO (Either String (D.Tensor,PixelFormat))
readImage file =
  I.readImage file >>= \case
    Left err -> return $ Left err
    Right img' -> return $ Right $ (fromDynImage img', pixelFormat img')

readImageAsRGB8 :: FilePath -> IO (Either String D.Tensor)
readImageAsRGB8 file =
  I.readImage file >>= \case
    Left err -> return $ Left err
    Right img' -> return . Right . fromDynImage . I.ImageRGB8 . I.convertRGB8 $ img'

pixelFormat :: I.DynamicImage -> PixelFormat
pixelFormat image = case image of
  I.ImageY8     _ -> Y8
  I.ImageYF     _ -> YF
  I.ImageYA8    _ -> YA8
  I.ImageRGB8   _ -> RGB8
  I.ImageRGBF   _ -> RGBF
  I.ImageRGBA8  _ -> RGBA8
  I.ImageYCbCr8 _ -> YCbCr8
  I.ImageCMYK8  _ -> CMYK8
  I.ImageCMYK16 _ -> CMYK16
  I.ImageRGBA16 _ -> RGBA16
  I.ImageRGB16  _ -> RGB16
  I.ImageY16    _ -> Y16
  I.ImageYA16   _ -> YA16
  I.ImageY32    _ -> Y32

fromDynImage :: I.DynamicImage -> D.Tensor
fromDynImage image = unsafePerformIO $ case image of
  I.ImageY8     (I.Image width height vec) -> createTensor width height 1 D.UInt8 1 vec
  I.ImageYF     (I.Image width height vec) -> createTensor width height 1 D.Float 4 vec
  I.ImageYA8    (I.Image width height vec) -> createTensor width height 2 D.UInt8 1 vec
  I.ImageRGB8   (I.Image width height vec) -> createTensor width height 3 D.UInt8 1 vec
  I.ImageRGBF   (I.Image width height vec) -> createTensor width height 3 D.Float 4 vec
  I.ImageRGBA8  (I.Image width height vec) -> createTensor width height 4 D.UInt8 1 vec
  I.ImageYCbCr8 (I.Image width height vec) -> createTensor width height 3 D.UInt8 1 vec
  I.ImageCMYK8  (I.Image width height vec) -> createTensor width height 4 D.UInt8 1 vec
  I.ImageCMYK16 (I.Image width height vec) -> createTensorU16to32 width height 4 D.Int32 vec
  I.ImageRGBA16 (I.Image width height vec) -> createTensorU16to32 width height 4 D.Int32 vec
  I.ImageRGB16  (I.Image width height vec) -> createTensorU16to32 width height 3 D.Int32 vec
  I.ImageY16    (I.Image width height vec) -> createTensorU16to32 width height 1 D.Int32 vec
  I.ImageYA16   (I.Image width height vec) -> createTensorU16to32 width height 2 D.Int32 vec
  I.ImageY32    (I.Image width height vec) -> createTensorU32to64 width height 1 D.Int64 vec
  where
    createTensor width height channel dtype dtype_size vec = do
      t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor) [1, height, width, channel] $ D.withDType dtype D.defaultOpts
      D.withTensor t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = width * height * channel * dtype_size
        F.withForeignPtr fptr $ \ptr2 -> do
          BSI.memcpy (F.castPtr ptr1) (F.castPtr ptr2) whc
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
  t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor) [num_imgs,height,width,channel] $ D.withDType D.UInt8 D.defaultOpts
  D.withTensor t $ \ptr1 -> do
    forM_ (zip [0..] imgs) $ \(idx,(I.Image width' height' vec)) -> do
      let (fptr,len) = V.unsafeToForeignPtr0 vec
          whc = width * height * channel
      when (len /= whc) $ do
        throwIO $ userError "vector's length is not the same as tensor' one."
      when (width /= width') $ do
        throwIO $ userError "image's width is not the same as first image's one"
      when (height /= height') $ do
        throwIO $ userError "image's height is not the same as first image's one"
      F.withForeignPtr fptr $ \ptr2 -> do
        BSI.memcpy (F.plusPtr (F.castPtr ptr1) (whc*idx)) ptr2 len
  return t

writeImage :: forall p. I.Pixel p => Int -> Int -> Int -> p -> D.Tensor -> IO (I.Image p)
writeImage width height channel pixel tensor = do
  let img@(I.Image w h vec) = I.generateImage (\_ _ -> pixel) width height :: I.Image p
  D.withTensor tensor $ \ptr1 -> do
    let (fptr,len) = V.unsafeToForeignPtr0 vec
        whc = width * height * channel
    if (len /= whc) then
      throwIO $ userError  $ "vector's length(" ++ show len ++ ") is not the same as tensor' one."
    else do
      F.withForeignPtr fptr $ \ptr2 -> do
        BSI.memcpy (F.castPtr ptr2) (F.castPtr ptr1) len
        return img

writeBitmap :: FilePath -> D.Tensor -> IO ()
writeBitmap file tensor = do
  case (D.shape tensor,D.dtype tensor) of
    ([1,height,width,1],D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([1,height,width],  D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([1,height,width,3],D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writeBitmap file
    ([height,width,1],D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([height,width],  D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writeBitmap file
    ([height,width,3],D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writeBitmap file
    format -> throwIO $ userError $ "Can not write a image. " ++ show format ++ " is unsupported format."

writePng :: FilePath -> D.Tensor -> IO ()
writePng file tensor = do
  case (D.shape tensor,D.dtype tensor) of
    ([1,height,width,1],D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([1,height,width],  D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([1,height,width,3],D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writePng file
    ([height,width,1],D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([height,width],  D.UInt8) -> writeImage width height 1 (0 :: I.Pixel8) tensor >>= I.writePng file
    ([height,width,3],D.UInt8) -> writeImage width height 3 (I.PixelRGB8 0 0 0) tensor >>= I.writePng file
    format -> throwIO $ userError $ "Can not write a image. " ++ show format ++ " is unsupported format."

-- [batch, height, width, channel] -> [batch, channel, height, width]
hwc2chw :: D.Tensor -> D.Tensor
hwc2chw = D.permute [0,3,1,2]

-- [batch, channel, height, width] -> [batch, height, width, channel]
chw2hwc :: D.Tensor -> D.Tensor
chw2hwc = D.permute [0,2,3,1]

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123
