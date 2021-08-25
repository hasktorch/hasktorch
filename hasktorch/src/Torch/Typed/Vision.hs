{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.Vision where

import qualified Codec.Compression.GZip as GZip
import qualified Codec.Picture as I
import Control.Exception.Safe ( SomeException (..), throwIO, throw)
import Control.Monad (forM_, (>=>), foldM)
import Data.Proxy (Proxy (..))
import Data.Type.Equality ((:~:)(..))
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import qualified Data.ByteString.Lazy as BS.Lazy
import qualified Data.List as List (transpose, unfoldr)
import qualified Data.Vector.Storable as V
import qualified Foreign.ForeignPtr as F
import qualified Foreign.Ptr as F
import GHC.Exts (IsList (fromList), Item)
import GHC.TypeLits
import qualified Language.C.Inline as C
import System.Directory (listDirectory, doesFileExist, doesDirectoryExist)
import System.IO.Unsafe ( unsafePerformIO )
import qualified Torch.DType as D
import Torch.Data.Pipeline
import qualified Torch.Device as D
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Managed.Type.Tensor.Tensor1 as LibTorch
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Torch.Tensor as D
import qualified Torch.TensorOptions as D
import Torch.Typed.Aux
import Torch.Typed.Factories (ones, zeros, empty)
import Torch.HList
import Torch.Typed.Functional
import Torch.Typed.Tensor
import GHC.Exts (IsList(fromList, fromListN, toList))
import Data.Kind

C.include "<stdint.h>"

-- MNIST dataset

data MNIST (m :: Type -> Type) (device :: (D.DeviceType, Nat) ) (batchSize :: Nat) = MNIST { mnistData :: MnistData }

instance (KnownNat batchSize, KnownDevice device, Applicative m) =>
  Dataset m (MNIST m device batchSize) Int (Tensor device 'D.Float '[batchSize, 784], Tensor device 'D.Int64 '[batchSize]) where
  getItem MNIST{..} ix =  
    let
      batchSize = natValI @batchSize
      indexes = [ix * batchSize .. (ix+1) * batchSize - 1]
      imgs =  getImages @batchSize mnistData indexes
      labels =  getLabels @batchSize mnistData indexes
    in pure (toDevice @device imgs, toDevice @device labels)

  keys MNIST{..} = fromList [ 0 .. Torch.Typed.Vision.length mnistData `Prelude.div` (natValI @batchSize) - 1]

data MnistData
  = MnistData
      { images :: BS.ByteString,
        labels :: BS.ByteString
      }

type Rows = 28

type Cols = 28

type DataDim = Rows * Cols

type ClassDim = 10

getLabels ::
  forall n. KnownNat n => MnistData -> [Int] -> CPUTensor 'D.Int64 '[n]
getLabels mnist imageIdxs =
  UnsafeMkTensor . D.asTensor . map (getLabel mnist) . take (natValI @n) $ imageIdxs

getLabel :: MnistData -> Int -> Int
getLabel mnist imageIdx =
  fromIntegral $ BS.index (labels mnist) (fromIntegral imageIdx + 8)

getImage :: MnistData -> Int -> CPUTensor 'D.Float '[DataDim]
getImage mnist imageIdx =
  let imageBS =
        [ fromIntegral $
            BS.index
              (images mnist)
              (fromIntegral imageIdx * 28 ^ 2 + 16 + r)
          | r <- [0 .. 28 ^ 2 - 1]
        ] ::
          [Float]
      (tensor :: CPUTensor 'D.Float '[DataDim]) =
        UnsafeMkTensor $ D.asTensor imageBS
   in tensor

getImages' ::
  forall n.
  KnownNat n =>
  MnistData ->
  [Int] ->
  CPUTensor 'D.Float '[n, DataDim]
getImages' mnist imageIdxs =
  UnsafeMkTensor $ D.asTensor $ map image $
    take
      (natValI @n)
      imageIdxs
  where
    image idx =
      [ fromIntegral $
          BS.index (images mnist) (fromIntegral idx * 28 ^ 2 + 16 + r)
        | r <- [0 .. 28 ^ 2 - 1]
      ] ::
        [Float]

getImages ::
  forall n.
  KnownNat n =>
  MnistData ->
  [Int] ->
  CPUTensor 'D.Float '[n, DataDim]
getImages mnist imageIdxs = UnsafeMkTensor $ unsafePerformIO $ do
  let (BSI.PS fptr off len) = images mnist
  t <-
    (cast2 LibTorch.empty_lo :: [Int] -> D.TensorOptions -> IO D.Tensor)
      [natValI @n, natValI @DataDim]
      (D.withDType D.UInt8 D.defaultOpts)
  D.withTensor t $ \ptr1 -> do
    F.withForeignPtr fptr $ \ptr2 -> do
      forM_ (zip [0 .. ((natValI @n) -1)] imageIdxs) $ \(i, idx) -> do
        BSI.memcpy
          (F.plusPtr ptr1 ((natValI @DataDim) * i))
          (F.plusPtr ptr2 (off + 16 + (natValI @DataDim) * idx))
          (natValI @DataDim)
  return $ D.toType D.Float t

length :: MnistData -> Int
length mnist = fromIntegral $ BS.length (labels mnist) - 8

decompressFile :: String -> String -> IO BS.ByteString
decompressFile path file = decompress' <$> BS.readFile (path <> "/" <> file)
  where
    decompress' = BS.concat . BS.Lazy.toChunks . GZip.decompress . BS.Lazy.fromStrict

initMnist :: String -> IO (MnistData, MnistData)
initMnist path = do
  imagesBS <- decompressFile path "train-images-idx3-ubyte.gz"
  labelsBS <- decompressFile path "train-labels-idx1-ubyte.gz"
  testImagesBS <- decompressFile path "t10k-images-idx3-ubyte.gz"
  testLabelsBS <- decompressFile path "t10k-labels-idx1-ubyte.gz"
  return (MnistData imagesBS labelsBS, MnistData testImagesBS testLabelsBS)


-- ImageFolder dataset 

newtype ImageFolder (m :: Type -> Type) (device :: (D.DeviceType, Nat) ) (finalShape :: [Nat]) = ImageFolder { folderData :: FolderData }


instance (KnownShape shape, All KnownNat shape, [n, 3, h, w] ~ shape, 1 <= n, KnownDevice device, Applicative m) => 
  Dataset m (ImageFolder m device shape) Int (Tensor device 'D.Float shape, Tensor device 'D.Int64 '[n]) where
  getItem ImageFolder{..} ix =  
    let
      [n, c, h, w] = shapeVal @shape
      indexes = [ix * n .. (ix+1) * n - 1]
      (imgs, labels) = case someShape indexes of
              SomeShape (Proxy :: Proxy idxs) -> 
                (getFolderImages @idxs folderData, 
                 getFolderLabels @idxs folderData)
    in pure (imgs, labels)

  keys ImageFolder{..} = fromList [ 0 .. (sum . map Prelude.length . imageNames $ folderData) `Prelude.div` (natValI @n) - 1]

getFolderImage :: 
  forall shape dtype device w h.
  (All KnownNat shape, KnownDevice device, KnownDType dtype,
  '[1, 3, w, h] ~ shape) =>
  FilePath ->
  [FilePath] -> 
  Int ->
  IO (Tensor device dtype shape)
getFolderImage path filePaths imgIdx = (readImageAsRGB8 @shape @dtype @device 
                                      . (++) path
                                      . (!!) filePaths
                                      $ (imgIdx `Prelude.mod` Prelude.length filePaths)) >>= \case  
                                          Left err -> throwIO . userError $ "Path contains non-image files"
                                          Right tensor -> return tensor


getFolderImages ::
  forall idxs shape dtype device n w h.
  (All KnownNat shape, KnownDType dtype, KnownDevice device,
   KnownShape idxs,
   1 <= n,
  '[n, 3, w, h] ~ shape) =>
  FolderData ->
  Tensor device dtype shape
getFolderImages folder = squeezeDim @0
                            . stack' @1 @'[1, 3, w, h] @(1 ': shape)
                            . concat
                            $ zipWith3 (\path names idxs -> map (unsafePerformIO . getFolderImage @'[1, 3, w, h] path names) idxs) 
                                       (sourcePaths folder)
                                       (imageNames folder)
                                       (List.transpose
                                          . takeWhile (not . null) . List.unfoldr (Just . splitAt (Prelude.length $ sourcePaths folder)) 
                                          $ shapeVal @idxs)

getFolderLabels ::
  forall idxs shape dtype device n.
  (KnownShape idxs, KnownShape shape, KnownDevice device, KnownDType dtype, All KnownNat shape,
   '[n] ~ shape
  ) =>
  FolderData ->
  Tensor device dtype shape 
getFolderLabels folder = stack' @0 @'[] @'[n]
                         . concat
                         $ zipWith (\a b -> map (\_ -> addScalar a (zeros :: Tensor device dtype '[])) b)
                                   (sourceLabels folder)
                                   (List.transpose
                                        . takeWhile (not . null) . List.unfoldr (Just . splitAt (Prelude.length $ sourcePaths folder)) 
                                        $ shapeVal @idxs)
  

 
data FolderData = FolderData {sourceLabels :: [Int], sourcePaths :: [FilePath], imageNames :: [[FilePath]]}

getImageFolder :: String -> IO FolderData
getImageFolder path' = do
     zippedPaths <- expandTree path'
     let (paths, contents) = unzip zippedPaths
     return $ FolderData (reverse [0..(Prelude.length paths - 1)]) paths contents
     where
        expandTree' :: FilePath -> [(FilePath, [FilePath])] -> FilePath -> IO [(FilePath, [FilePath])]
        expandTree' currentPath currentList path = expandTree (currentPath ++ path) >>= return . (++ currentList)

        expandTree :: FilePath -> IO [(FilePath, [FilePath])]
        expandTree currentPath = do
              let path = if last currentPath == '/' then currentPath else currentPath ++ "/" 
              a <- listDirectory path :: IO [FilePath]
              b <- doesFileExist . (++) path . head $ a
              c <- doesDirectoryExist . (++) path . head $ a
              let expand' | b = return [(path, a)] :: IO [(FilePath, [FilePath])]
                          | c && not b = foldM (expandTree' path) [] a
                          | otherwise  = error "Invalid file type in folder tree"
              expand'
            
            
            
              

-- Special implementation of SomeShape for working with elements of 2D tensors

data SomeShape2d where
  SomeShape2d :: 
    forall (shape :: [Nat]) (n :: Nat) (c :: Nat) (h :: Nat) (w :: Nat). 
    (  KnownShape shape
     , All KnownNat shape
     , (n ': h ': c ': w ': '[]) ~ shape) => 
    Proxy shape -> 
    SomeShape2d

someShape2d :: [Int] -> SomeShape2d
someShape2d [] = error "Invalid"
someShape2d [n, h, c, w] = case map (someNatVal . fromIntegral) [n, h, c, w] of
  [Just (SomeNat (Proxy :: Proxy n')),
   Just (SomeNat (Proxy :: Proxy h')),
   Just (SomeNat (Proxy :: Proxy c')),
   Just (SomeNat (Proxy :: Proxy w'))] -> SomeShape2d $ Proxy @'[n', h', c', w']

-- Typed image/image tensor processing

fromDynImage :: 
  forall outShape dtype device c h w. 
  ( KnownShape outShape, All KnownNat outShape,  
    KnownDType dtype, KnownDevice device,
    outShape ~ [1, c, h, w]
  ) => 
  I.DynamicImage -> 
  Tensor device dtype outShape
fromDynImage image = unsafePerformIO $ case image of
  I.ImageY8 (I.Image width height vec) -> case someShape2d [1, height, width, 1] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 1) of
                                                Just Refl -> createTensor @'[1, h', w', 1] @outShape @dtype @'D.UInt8 @device @1 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageYF (I.Image width height vec) -> case someShape2d [1, height, width, 1] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 1) of
                                                Just Refl -> createTensor @'[1, h', w', 1] @outShape @dtype @'D.Float @device @4 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageYA8 (I.Image width height vec) -> case someShape2d [1, height, width, 2] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 2) of
                                                Just Refl -> createTensor @'[1, h', w', 2] @outShape @dtype @'D.UInt8 @device @1 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageRGB8 (I.Image width height vec) -> case someShape2d [1, height, width, 3] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 3) of
                                                Just Refl -> createTensor @'[1, h', w', 3] @outShape @dtype @'D.UInt8 @device @1 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageRGBF (I.Image width height vec) -> case someShape2d [1, height, width, 3] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 3) of
                                                Just Refl -> createTensor @'[1, h', w', 3] @outShape @dtype @'D.Float @device @4 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageRGBA8 (I.Image width height vec) -> case someShape2d [1, height, width, 4] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 4) of
                                                Just Refl -> createTensor @'[1, h', w', 4] @outShape @dtype @'D.UInt8 @device @1 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageYCbCr8 (I.Image width height vec) -> case someShape2d [1, height, width, 3] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 3) of
                                                Just Refl -> createTensor @'[1, h', w', 3] @outShape @dtype @'D.UInt8 @device @1 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageCMYK8 (I.Image width height vec) -> case someShape2d [1, height, width, 4] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 4) of
                                                Just Refl -> createTensor @'[1, h', w', 4] @outShape @dtype @'D.UInt8 @device @1 vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageCMYK16 (I.Image width height vec) -> case someShape2d [1, height, width, 4] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 4) of
                                                Just Refl -> createTensorU16to32 @'[1, h', w', 4] @outShape @dtype @'D.Int32 @device vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageRGBA16 (I.Image width height vec) -> case someShape2d [1, height, width, 4] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 4) of
                                                Just Refl -> createTensorU16to32 @'[1, h', w', 4] @outShape @dtype @'D.Int32 @device vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageRGB16 (I.Image width height vec) -> case someShape2d [1, height, width, 3] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 3) of
                                                Just Refl -> createTensorU16to32 @'[1, h', w', 3] @outShape @dtype @'D.Int32 @device vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageY16 (I.Image width height vec) -> case someShape2d [1, height, width, 1] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 1) of
                                                Just Refl -> createTensorU16to32 @'[1, h', w', 1] @outShape @dtype @'D.Int32 @device vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageYA16 (I.Image width height vec) -> case someShape2d [1, height, width, 2] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 2) of
                                                Just Refl -> createTensorU16to32 @'[1, h', w', 2] @outShape @dtype @'D.Int32 @device vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
  I.ImageY32 (I.Image width height vec) -> case someShape2d [1, height, width, 1] of
                                            SomeShape2d (_ :: Proxy '[n', h', w', c']) -> 
                                              case sameNat (Proxy :: Proxy c) (Proxy :: Proxy 1) of
                                                Just Refl -> createTensorU32to64 @'[1, h', w', 1] @outShape @dtype @'D.Int64 @device vec
                                                Nothing -> error "Incorrect Channel Number in output shape."
                                            
  where
    createTensor :: 
      forall shape outShape outDtype dtype' device dtype_size n' c' h' w' n0 c0 h0 w0 vec. 
      (KnownShape shape,  All KnownNat shape,
       KnownShape outShape, All KnownNat outShape,
       KnownDType outDtype, KnownDType dtype', KnownDevice device, KnownNat dtype_size, V.Storable vec,
       (n' ': c' ': h' ': w' ': '[]) ~ outShape,
       (n0 ': h0 ': w0 ': c0 ': '[]) ~ shape,
       n' ~ n0,
       c' ~ c0
      ) => 
      V.Vector vec -> 
      IO (Tensor device outDtype outShape)  
    createTensor vec = do
      let [_, h, w, c] = shapeVal @shape
          dtype_size = natValI @dtype_size
      t <- empty @shape @dtype' @'(D.CPU, 0)
      withTensorPtr t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = w * h * c * dtype_size
        F.withForeignPtr fptr $ \ptr2 -> do
          BSI.memcpy (F.castPtr ptr1) (F.castPtr ptr2) whc
          print t
          return $ toDevice @device @'(D.CPU, 0)
                    . toDType @outDtype @dtype'
                    . resize2d @outShape @"bilinear" True 
                    . hwc2chw @'(D.CPU, 0) @dtype' @shape
                    $ t 
    createTensorU16to32 :: 
      forall shape outShape outDtype dtype' device n' c' h' w' n0 c0 h0 w0 vec. 
        (KnownShape shape, All KnownNat shape, 
         KnownShape outShape, All KnownNat outShape,
         KnownDType outDtype, KnownDType dtype', KnownDevice device, V.Storable vec,
         (n' ': c' ': h' ': w' ': '[]) ~ outShape,
         (n0 ': h0 ': w0 ': c0 ': '[]) ~ shape,
         n' ~ n0,
         c' ~ c0
        ) =>  
        V.Vector vec -> 
        IO (Tensor device outDtype outShape)
    createTensorU16to32 vec = do
      let [_, h, w, c] = shapeVal @shape
      t <- empty @shape @dtype' @'(D.CPU, 0)
      withTensorPtr t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = fromIntegral $ w * h * c
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
          return $ toDevice @device @'(D.CPU, 0)
                    . toDType @outDtype @dtype'
                    . resize2d @outShape @"bilinear" True 
                    . hwc2chw @'(D.CPU, 0) @dtype' @shape
                    $ t 
    createTensorU32to64 :: 
      forall shape outShape outDtype dtype' device n' c' h' w' n0 c0 h0 w0 vec. 
        (KnownShape shape, All KnownNat shape, 
         KnownShape outShape, All KnownNat outShape,
         KnownDType outDtype, KnownDType dtype', KnownDevice device, V.Storable vec,
         (n' ': c' ': h' ': w' ': '[]) ~ outShape,
         (n0 ': h0 ': w0 ': c0 ': '[]) ~ shape,
         n' ~ n0,
         c' ~ c0
        ) =>
        V.Vector vec -> 
        IO (Tensor device outDtype outShape)
    createTensorU32to64 vec = do
      let [_, h, w, c] = shapeVal @shape
      t <- empty @shape @dtype' @'(D.CPU, 0)
      withTensorPtr t $ \ptr1 -> do
        let (fptr, len) = V.unsafeToForeignPtr0 vec
            whc = fromIntegral $ w * h * c
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
          return $ toDevice @device @'(D.CPU, 0)
                    . toDType @outDtype @dtype'
                    . resize2d @outShape @"bilinear" True 
                    . hwc2chw @'(D.CPU, 0) @dtype' @shape
                    $ t 

readImageAsRGB8 :: 
  forall shape dtype device c h w.
  (KnownShape shape, All KnownNat shape,
   KnownDType dtype, KnownDevice device,
   [1, 3, h, w] ~ shape) => 
  FilePath -> 
  IO (Either String (Tensor device dtype shape))
readImageAsRGB8 file =
  I.readImage file >>= \case
    Left err -> return $ Left err
    Right img' -> return . Right . fromDynImage @shape @dtype @device . I.ImageRGB8 . I.convertRGB8 $ img'

writeImage :: 
  forall shape' p shape dtype device. 
  (I.Pixel p,
   KnownShape shape'
  ) => 
  p -> Tensor device dtype shape -> IO (I.Image p)
writeImage pixel tensor = do
  let [n, h, w, c] = shapeVal @shape'
      img@(I.Image w' h' vec) = I.generateImage (\_ _ -> pixel) w h :: I.Image p
  withTensorPtr tensor $ \ptr1 -> do
    let (fptr, len) = V.unsafeToForeignPtr0 vec
        whc = w * h * c
    if (len /= whc)
      then throwIO $ userError $ "vector's length(" ++ show len ++ ") is not the same as tensor' one."
      else do
        F.withForeignPtr fptr $ \ptr2 -> do
          BSI.memcpy (F.castPtr ptr2) (F.castPtr ptr1) len
          return img


writePng :: 
 forall shape dtype device. 
  (KnownShape shape, All KnownNat shape,
   KnownShape (shape ++ '[1]),
   2 <= ListLength shape,
   ListLength shape <= 4,
   KnownDType dtype,
   'D.UInt8 ~ dtype
  ) => 
  FilePath -> Tensor device dtype shape -> IO ()
writePng file tensor = do
  case shapeVal @shape of
    [1, height, width, 1] -> writeImage @shape (0 :: I.Pixel8) tensor >>= I.writePng file
    [1, height, width] -> writeImage @(shape ++ '[1]) (0 :: I.Pixel8) tensor >>= I.writePng file
    [1, height, width, 3] -> writeImage @shape (I.PixelRGB8 0 0 0) tensor >>= I.writePng file
    [height, width, 1] -> writeImage @(1 ': shape) (0 :: I.Pixel8) tensor >>= I.writePng file
    [height, width] -> writeImage @(1 ': (shape ++ '[1])) (0 :: I.Pixel8) tensor >>= I.writePng file
    [height, width, 3] -> writeImage @(1 ': shape) (I.PixelRGB8 0 0 0) tensor >>= I.writePng file
    format -> throwIO $ userError $ "Can not write a image. " ++ show format ++ " is unsupported format."

-- Other permute checks can be bypassed with known dimensions

-- [batch, height, width, channel] -> [batch, channel, height, width]
hwc2chw ::   
  forall device dtype shape shape'.
  ( 4 ~ ListLength shape,
    shape' ~ PermuteDims shape shape '[0, 3, 1, 2] 0
  ) =>
  Tensor device dtype shape ->
  Tensor device dtype shape' -- output
hwc2chw t = unsafePerformIO $ cast2 LibTorch.tensor_permute_l t ([0, 3, 1, 2] :: [Int])

-- [batch, channel, height, width] -> [batch, height, width, channel]
chw2hwc ::   
  forall device dtype shape shape'.
  ( 4 ~ ListLength shape,
    shape' ~ PermuteDims shape shape '[0, 2, 3, 1] 0
  ) =>
  Tensor device dtype shape ->
  Tensor device dtype shape' -- output
chw2hwc t = unsafePerformIO $ cast2 LibTorch.tensor_permute_l t ([0, 2, 3, 1] :: [Int])


