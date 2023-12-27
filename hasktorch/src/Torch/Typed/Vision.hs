{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.Vision where

import qualified Codec.Compression.GZip as GZip
import Control.Monad (forM_)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import Foreign.Marshal.Utils (copyBytes)
import qualified Data.ByteString.Lazy as BS.Lazy
import Data.Kind
import qualified Foreign.ForeignPtr as F
import qualified Foreign.Ptr as F
import GHC.Exts (IsList (fromList))
import GHC.TypeLits
import System.IO.Unsafe
import qualified Torch.DType as D
import Torch.Data.Pipeline
import qualified Torch.Device as D
import Torch.Internal.Cast
import qualified Torch.Internal.Managed.TensorFactories as LibTorch
import qualified Torch.Tensor as D
import qualified Torch.TensorOptions as D
import Torch.Typed.Auxiliary
import Torch.Typed.Functional
import Torch.Typed.Tensor

data MNIST (m :: Type -> Type) (device :: (D.DeviceType, Nat)) (batchSize :: Nat) = MNIST {mnistData :: MnistData}

instance
  (KnownNat batchSize, KnownDevice device, Applicative m) =>
  Dataset m (MNIST m device batchSize) Int (Tensor device 'D.Float '[batchSize, 784], Tensor device 'D.Int64 '[batchSize])
  where
  getItem MNIST {..} ix =
    let batchSize = natValI @batchSize
        indexes = [ix * batchSize .. (ix + 1) * batchSize - 1]
        imgs = getImages @batchSize mnistData indexes
        labels = getLabels @batchSize mnistData indexes
     in pure (toDevice @device imgs, toDevice @device labels)

  keys MNIST {..} = fromList [0 .. Torch.Typed.Vision.length mnistData `Prelude.div` (natValI @batchSize) - 1]

data MnistData = MnistData
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
  UnsafeMkTensor $
    D.asTensor $
      map image $
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
getImages mnist imageIdxs = UnsafeMkTensor $
  unsafePerformIO $ do
    let (BSI.PS fptr off len) = images mnist
    t <-
      (cast2 LibTorch.empty_lo :: [Int] -> D.TensorOptions -> IO D.Tensor)
        [natValI @n, natValI @DataDim]
        (D.withDType D.UInt8 D.defaultOpts)
    D.withTensor t $ \ptr1 -> do
      F.withForeignPtr fptr $ \ptr2 -> do
        forM_ (zip [0 .. ((natValI @n) -1)] imageIdxs) $ \(i, idx) -> do
          copyBytes
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
