{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
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
import Data.Finite (Finite, packFinite)
import Data.Functor.Const (Const (..))
import Data.List (sortOn)
import Data.Maybe (fromJust)
import Data.Ord (Down (..))
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Storable as VS
import Foreign.Marshal.Utils (copyBytes)
import qualified Data.ByteString.Lazy as BS.Lazy
import Data.Kind
import GHC.Generics (Generic)
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
import Torch.Lens (Lens')
import Torch.Typed.Auxiliary
import Torch.Typed.Functional
import Torch.Typed.Lens (field)
import Torch.Typed.Staged (Cond, maxE, minE)
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

--------------------------------------------------------------------------------
-- Non-maximum suppression
--------------------------------------------------------------------------------

-- | A detection: box coordinates and a confidence score, addressed by name.
-- Used as a named dimension, so detections have type
-- @NamedTensor device dtype '[Vector n, Box]@ and nothing below indexes a
-- coordinate by number.
data Box a = Box
  { x1 :: a,
    y1 :: a,
    x2 :: a,
    y2 :: a,
    score :: a
  }
  deriving (Show, Eq, Generic, Functor)

-- | Intersection over union of two boxes, as a formula on scalars.
--
-- Written against @(Fractional a, Cond a)@ so that the one definition serves
-- both as the vectorized implementation ('boxIou' instantiates it at
-- @a = Tensor@) and as a per-element reference (at @a = Float@); the test
-- suite checks the two instantiations agree exactly.
iou :: (Fractional a, Cond a) => Box a -> Box a -> a
iou a b = inter / (area a + area b - inter)
  where
    iw = maxE 0 (minE (x2 a) (x2 b) - maxE (x1 a) (x1 b))
    ih = maxE 0 (minE (y2 a) (y2 b) - maxE (y1 a) (y1 b))
    inter = iw * ih
    area v = (x2 v - x1 v) * (y2 v - y1 v)

-- | The pairwise IoU matrix of a set of detections: 'iou' evaluated once at
-- @a = Tensor@, with the coordinate fields shaped as a column and a row so
-- broadcasting produces the whole \(n \times n\) matrix.
boxIou ::
  forall n device.
  KnownNat n =>
  NamedTensor device 'D.Float '[Vector n, Box] ->
  NamedTensor device 'D.Float '[Vector n, Vector n]
boxIou dets = fromUnnamed . UnsafeMkTensor $ iou rows cols
  where
    n = natValI @n
    rows = D.reshape [n, 1] <$> boxFields dets
    cols = D.reshape [1, n] <$> boxFields dets

-- | The fields of all detections at once, each as a plain @[n]@ tensor,
-- extracted via the named-field lenses.
boxFields ::
  forall n device.
  NamedTensor device 'D.Float '[Vector n, Box] ->
  Box D.Tensor
boxFields dets =
  Box
    (viewL (field @"x1"))
    (viewL (field @"y1"))
    (viewL (field @"x2"))
    (viewL (field @"y2"))
    (viewL (field @"score"))
  where
    viewL ::
      Lens'
        (NamedTensor device 'D.Float '[Vector n, Box])
        (NamedTensor device 'D.Float '[Vector n]) ->
      D.Tensor
    viewL l = toDynamic (getConst (l Const dets))

-- | Greedy non-maximum suppression: take the best-scoring box, drop every
-- remaining box whose IoU with it exceeds the threshold, recurse.  Returns
-- indices into the input, best score first.
--
-- Only the IoU rows of boxes that are actually kept get computed: 'iou' is
-- evaluated once per kept box — the box's fields as scalars, broadcast
-- against all boxes at once — so memory stays @O(n)@ and no @n^2@ matrix is
-- materialized.  ('boxIou' is there when the full matrix is wanted.)  The
-- suppression itself is plain list recursion, which is where the algorithm
-- is easiest to read.
nms ::
  forall n device.
  KnownNat n =>
  -- | IoU threshold
  Float ->
  -- | detections
  NamedTensor device 'D.Float '[Vector n, Box] ->
  -- | kept indices, best score first
  [Finite n]
nms threshold dets = map (fromJust . packFinite . fromIntegral) (go order)
  where
    go [] = []
    go (i : rest) = i : go [j | j <- rest, row VS.! j <= threshold]
      where
        row = iouRow i
    -- IoU of box i against every box: the same scalar formula, its left
    -- argument a box of 0-dim tensors broadcast against [n] tensors
    iouRow i = D.asValue (cpu (iou (D.select 0 i <$> allBoxes) allBoxes)) :: VS.Vector Float
    allBoxes = boxFields dets
    order = map snd (sortOn (Down . fst) (zip (VS.toList scores) [0 ..]))
    scores = D.asValue (cpu (score allBoxes)) :: VS.Vector Float
    cpu = D.toDevice (D.Device D.CPU 0)
