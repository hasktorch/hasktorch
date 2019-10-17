{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}

module Image where

import           Control.Monad (forM_)
import           System.IO.Unsafe
import qualified Codec.Compression.GZip        as GZip
import qualified Data.ByteString               as BS
import qualified Data.ByteString.Lazy          as BS.Lazy
import qualified Data.ByteString.Internal      as BSI
import           GHC.TypeLits

import           ATen.Cast
import           Torch.Static
import           Torch.Static.Native
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorOptions           as D
import qualified Foreign.ForeignPtr            as F
import qualified Foreign.Ptr                   as F
import qualified Torch.Managed.Native          as LibTorch



data MnistData =
  MnistData
  { images :: BS.ByteString
  , labels :: BS.ByteString
  }

type Rows = 28
type Cols = 28
type DataDim = Rows * Cols
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

getImages'
  :: forall n
   . KnownNat n
  => MnistData
  -> [Int]
  -> Tensor 'D.Float '[n, DataDim]
getImages' mnist imageIdxs = UnsafeMkTensor $ D.asTensor $ map image $ take
  (natValI @n)
  imageIdxs
 where
  image idx =
    [ fromIntegral
        $ BS.index (images mnist) (fromIntegral idx * 28 ^ 2 + 16 + r)
    | r <- [0 .. 28 ^ 2 - 1]
    ] :: [Float]

getImages
  :: forall n
   . KnownNat n
  => MnistData
  -> [Int]
  -> Tensor 'D.Float '[n, DataDim]
getImages mnist imageIdxs = UnsafeMkTensor $ unsafePerformIO $ do
  let (BSI.PS fptr off len) = images mnist
  t <- ((cast2 LibTorch.empty_lo) :: [Int] -> D.TensorOptions -> IO D.Tensor)
         [(natValI @n),(natValI @DataDim)]
         (D.withDType D.UInt8 D.defaultOpts)
  D.withTensor t $ \ptr1 -> do
    F.withForeignPtr fptr $ \ptr2 -> do
      forM_ (zip [0..((natValI @n)-1)] imageIdxs) $ \(i,idx) -> do
        BSI.memcpy
          (F.plusPtr ptr1 ((natValI @DataDim)*i))
          (F.plusPtr ptr2 (off+16+(natValI @DataDim)*idx))
          (natValI @DataDim)
  return $ D.toType D.Float t

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
