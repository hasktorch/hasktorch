module UntypedImage where

import Control.Monad (forM_)

import           ATen.Cast
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BSI
import qualified Torch.DType as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorOptions           as D
import qualified Foreign.ForeignPtr            as F
import qualified Foreign.Ptr                   as F
import qualified Torch.Managed.Native as LibTorch

import qualified Image as I
import Torch.Tensor
import Torch.NN

getLabels' :: Int -> I.MnistData -> [Int] -> Tensor
getLabels' n mnist imageIdxs =
  asTensor $ map (I.getLabel mnist) . take n $ imageIdxs

getImages ::
  Int
  -> Int
  -> I.MnistData
  -> [Int]
  -> IO Tensor
getImages n dataDim mnist imageIdxs = do
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
