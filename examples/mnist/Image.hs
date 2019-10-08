{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Image where

import Data.Int
import Data.Word
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS

-- Torch deps

import GHC.TypeLits

import Torch.Static
import Torch.Static.Native
import qualified Torch.DType as D
import qualified Torch.Tensor as D


data MnistData =
  MnistData
  { images :: BS.ByteString
  , labels :: BS.ByteString
  }

type DataDim = 784
type ClassDim = 10

getLabels :: forall n. KnownNat n => MnistData -> [Int] -> Tensor 'D.Float '[n,ClassDim]
getLabels mnist imageIdxs = UnsafeMkTensor $ D.asTensor $ map (getLabel mnist) $ take (natValI @n) imageIdxs

getLabel :: MnistData -> Int -> [Float]
getLabel mnist imageIdx =
  let v = fromIntegral $ BS.index (labels mnist) ((fromIntegral imageIdx) + 8) :: Word8
  in case v of
       1 -> [1,0,0,0,0,0,0,0,0,0]
       2 -> [0,1,0,0,0,0,0,0,0,0]
       3 -> [0,0,1,0,0,0,0,0,0,0]
       4 -> [0,0,0,1,0,0,0,0,0,0]
       5 -> [0,0,0,0,1,0,0,0,0,0]
       6 -> [0,0,0,0,0,1,0,0,0,0]
       7 -> [0,0,0,0,0,0,1,0,0,0]
       8 -> [0,0,0,0,0,0,0,1,0,0]
       9 -> [0,0,0,0,0,0,0,0,1,0]
       _ -> [0,0,0,0,0,0,0,0,0,1]

getImage :: MnistData -> Int -> Tensor 'D.Float '[DataDim]
getImage mnist imageIdx =
  let imageBS = [fromIntegral $ BS.index (images mnist) ((fromIntegral imageIdx) * 28^2 + 16 + r) | r <- [0..28^2 - 1]] :: [Float]
      (tensor :: Tensor 'D.Float '[DataDim]) = UnsafeMkTensor $ D.asTensor imageBS
  in tensor

getImages :: forall n. KnownNat n => MnistData -> [Int] -> Tensor 'D.Float '[n, DataDim]
getImages mnist imageIdxs = UnsafeMkTensor $ D.asTensor $ map image $ take (natValI @n) imageIdxs
  where
    image idx = [fromIntegral $ BS.index (images mnist) ((fromIntegral idx) * 28^2 + 16 + r) | r <- [0..28^2 - 1]] :: [Float]

length :: MnistData -> Int
length mnist = fromIntegral $ (BS.length (labels mnist)) - 8


initMnist :: IO (MnistData, MnistData)
initMnist = do
  let path = "data"
  imagesBS <- decompress <$> BS.readFile (path <>  "/" <> "train-images-idx3-ubyte.gz")
  labelsBS <- decompress <$> BS.readFile (path <>  "/" <> "train-labels-idx1-ubyte.gz")
  testImagesBS <- decompress <$> BS.readFile (path <>  "/" <> "t10k-images-idx3-ubyte.gz")
  testLabelsBS <- decompress <$> BS.readFile (path <>  "/" <> "t10k-labels-idx1-ubyte.gz")
  return (MnistData imagesBS labelsBS, MnistData testImagesBS testLabelsBS)

