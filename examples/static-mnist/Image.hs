{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Image where

import qualified Codec.Compression.GZip        as GZip
import qualified Data.ByteString               as BS
import qualified Data.ByteString.Lazy          as BS.Lazy
import           GHC.TypeLits

import           Torch.Static
import           Torch.Static.Native
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D


data MnistData =
  MnistData
  { images :: BS.ByteString
  , labels :: BS.ByteString
  }

type DataDim = 784
type ClassDim = 10

getLabels
  :: forall n . KnownNat n => MnistData -> [Int] -> Tensor 'D.Int64 '[n]
getLabels mnist imageIdxs =
  UnsafeMkTensor $ D.asTensor $ map (getLabel mnist) $ take (natValI @n)
                                                            imageIdxs

getLabel :: MnistData -> Int -> Int
getLabel mnist imageIdx =
  fromIntegral $ BS.index (labels mnist) (fromIntegral imageIdx + 8)

getImage :: MnistData -> Int -> Tensor 'D.Float '[DataDim]
getImage mnist imageIdx =
  let imageBS =
          [ fromIntegral $ BS.index (images mnist)
                                    (fromIntegral imageIdx * 28 ^ 2 + 16 + r)
          | r <- [0 .. 28 ^ 2 - 1]
          ] :: [Float]
      (tensor :: Tensor 'D.Float '[DataDim]) =
          UnsafeMkTensor $ D.asTensor imageBS
  in  tensor

getImages
  :: forall n
   . KnownNat n
  => MnistData
  -> [Int]
  -> Tensor 'D.Float '[n, DataDim]
getImages mnist imageIdxs = UnsafeMkTensor $ D.asTensor $ map image $ take
  (natValI @n)
  imageIdxs
 where
  image idx =
    [ fromIntegral
        $ BS.index (images mnist) (fromIntegral idx * 28 ^ 2 + 16 + r)
    | r <- [0 .. 28 ^ 2 - 1]
    ] :: [Float]

length :: MnistData -> Int
length mnist = fromIntegral $ BS.length (labels mnist) - 8

initMnist :: IO (MnistData, MnistData)
initMnist = do
  let path = "data"
      decompress' =
        BS.concat . BS.Lazy.toChunks . GZip.decompress . BS.Lazy.fromStrict
  imagesBS <- decompress'
    <$> BS.readFile (path <> "/" <> "train-images-idx3-ubyte.gz")
  labelsBS <- decompress'
    <$> BS.readFile (path <> "/" <> "train-labels-idx1-ubyte.gz")
  testImagesBS <- decompress'
    <$> BS.readFile (path <> "/" <> "t10k-images-idx3-ubyte.gz")
  testLabelsBS <- decompress'
    <$> BS.readFile (path <> "/" <> "t10k-labels-idx1-ubyte.gz")
  return (MnistData imagesBS labelsBS, MnistData testImagesBS testLabelsBS)
