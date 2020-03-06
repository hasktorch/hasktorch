{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}

module Torch.Utils.Image where
import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                , throwIO
                                                )
import           Foreign.ForeignPtr
import           System.IO.Unsafe

import qualified Torch.Internal.Managed.TensorFactories          as LibTorch
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functional               as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D

import qualified Codec.Picture                 as I
import           System.IO                     (FilePath)
import           Torch.Internal.Cast
import qualified Data.Vector.Storable          as V
import qualified Foreign.ForeignPtr            as F
import qualified Foreign.Ptr                   as F
import qualified Data.ByteString.Internal      as BSI

readImage :: FilePath -> IO (Either String D.Tensor)
readImage file = I.readImage file >>= image2tensor

image2tensor' :: forall p. I.Pixel p => I.Image p -> IO D.Tensor
image2tensor' img = do
  v <- image2tensor img
  case v of
    Left err -> throwIO $ userError err
    Right i -> return i

image2tensor :: forall p. I.Pixel p => I.Image p -> IO (Either String D.Tensor)
image2tensor dynamic_img =
  case dynamic_img of
    Left err -> return $ Left err
    Right img' -> fromDynImage img'

fromDynImage :: I.DynamicImage -> IO (Either String D.Tensor)
fromDynImage image = case image of
  I.ImageY8 (I.Image width height vec) -> createTensor width height 1 D.UInt8 vec
  I.ImageYA8 (I.Image width height vec) -> createTensor width height 2 D.UInt8 vec
  I.ImageRGB8 (I.Image width height vec) -> createTensor width height 3 D.UInt8 vec
  I.ImageRGBA8 (I.Image width height vec) -> createTensor width height 4 D.UInt8 vec
  I.ImageYCbCr8 (I.Image width height vec) -> createTensor width height 3 D.UInt8 vec
  I.ImageCMYK8 (I.Image width height vec) -> createTensor width height 4 D.UInt8 vec
  _ -> return $ Left "Unsupported format, convert to 8 bit Image first"
  where
    createTensor width height channel dtype vec = do
      t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor) [1, height, width, channel] $ D.withDType dtype D.defaultOpts
      D.withTensor t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = width * height * channel
        if (len /= whc)
          then return $ Left "vector's length is not the same as tensor' one."
          else do
            F.withForeignPtr fptr $ \ptr2 -> do
              BSI.memcpy (F.castPtr ptr1) ptr2 len
              return $ Right t

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
hwc2chw input = D.permute input [0,3,1,2]

-- [batch, channel, height, width] -> [batch, height, width, channel]
chw2hwc :: D.Tensor -> D.Tensor
chw2hwc input = D.permute input [0,2,3,1]
