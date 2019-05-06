{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TupleSections #-}

import Prelude as P

-- Dependencies
import Text.Printf (printf)
import Control.Monad (replicateM, when)
import Control.Monad.ST (ST, runST)
import Data.Maybe (fromJust)
import Numeric.Backprop (BVar, Reifies, W, (^^.), backprop, backprop2)
import System.IO (hFlush, stdout)
import qualified Data.Vector as V (fromList)
import Lens.Micro
import Data.List.Split
import Data.Int
import Data.Functor
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Control.Monad

import Numeric.Backprop as Bp

-- Torch deps
import qualified Torch.Core.Random as RNG
import Torch.Double as T
import Torch.Double
  ( Tensor, HsReal                    -- Torch.Sig.Types
  , unsafeMatrix, unsafeVector, get1d -- Torch.Indef.Static.Tensor
  , constant                          -- Torch.Indef.Static.Tensor.Math
  , (^*)                              -- Torch.Indef.Static.Pairwise
  , (.:)                              -- Torch.Indef.Types (helper function)
  , uniform, ord2Tuple                -- Torch.Indef.Math.Random.TH
  , positive, positiveValue, Positive -- Torch.Indef.Math.Random.TH
  , manualSeed, newRNG                -- Torch.Core.Random
  , eqTensor                          -- Torch.Indef.Static.CompareT
  , allOf                             -- Torch.Indef.Mask
  , mSECriterionIO                    -- Torch.Indef.Static.NN.Criterion
  )
import Torch.Double.NN.Linear (Linear(Linear), getTensors)
import qualified Torch.Double.NN.Criterion as Bp (mSECriterion)
import qualified Torch.Double.NN.Linear as Bp (linearBatch)
import qualified Torch.Double as Bp (relu)

type DataDim = 784
type BatchSize = 64
type MLP3 i h1 h2 o = (Linear i h1, Linear h1 h2, Linear h2 o)
type MnistArch = MLP3 DataDim 128 128 10

seedVal :: RNG.Seed
seedVal = 3141592653579


render :: HsReal -> Char
render n =
    s P.!! ((P.floor n) * length s `P.div` 256)
    where
        s = " .:oO@"


image2tensor :: Int64
             -> BS.ByteString
             -> IO (Tensor '[DataDim])
image2tensor imageIdx images = do
    let imageBS = [fromIntegral $ BS.index images (imageIdx * 28^2 + 16 + r) | r <- [0..28^2 - 1]]
    Just (tensor :: Tensor '[DataDim]) <- fromList imageBS
    pure tensor


genBatch :: RNG.Generator
         -> (BS.ByteString, BS.ByteString)
         -> Int
         -> IO ([(Tensor '[DataDim], Int)])
genBatch gen (images, labels) batchSize = do
    imageIdxs <- (fmap $ map $ P.round . (60000 *)) $ replicateM batchSize $ RNG.uniform gen 0 1
    let tensorIOs :: [IO (Tensor '[DataDim])] = [image2tensor imageIdx images | imageIdx <- imageIdxs]
    let tensorsIO :: IO ([Tensor '[DataDim]]) = sequence tensorIOs

    let lbls :: [Int] = [fromIntegral $ BS.index labels (imageIdx + 8) | imageIdx <- imageIdxs]

    tensors :: [Tensor '[DataDim]] <- tensorsIO
    pure $ zip tensors lbls


mkNetwork :: IO MnistArch
mkNetwork = do
  g <- newRNG
  let Just rg = ord2Tuple (0, 1)
  l1 <- fmap Linear $ (,) <$> uniform g rg <*> uniform g rg
  l2 <- fmap Linear $ (,) <$> uniform g rg <*> uniform g rg
  l3 <- fmap Linear $ (,) <$> uniform g rg <*> uniform g rg
  pure (l1, l2, l3)


forward
  :: Reifies s W
  => BVar s MnistArch
  -> BVar s (Tensor '[1, DataDim])
  -> BVar s (Tensor '[1, 10])
forward arch inp
  = Bp.linearBatch (arch ^^. _1) inp
  & T.relu
  & Bp.linearBatch (arch ^^. _2)
  & T.relu
  & Bp.linearBatch (arch ^^. _3)


updateNet
  :: MnistArch
  -> (Positive HsReal, MnistArch)
  -> MnistArch
updateNet (l1, l2, l3) (plr, (g1, g2, g3)) =
  (l1 - (g1 ^* lr), l2 - (g2 ^* lr), l3 - (g3 ^* lr))
    where
      lr = positiveValue plr

trainer
  :: Positive HsReal                   -- learning rate
  -> Int                               -- number of batches to generate
  -> RNG.Generator
  -> (BS.ByteString, BS.ByteString)
  -> MnistArch                         -- our initial network
  -> IO MnistArch                      -- our final, trained network
trainer lr n gen (images, labels) net0 = go 0 net0
  where
    go c net
      | c >= n = pure net
      | otherwise = do
        batch <- genBatch gen (images, labels) 1
        let (image, lbl) = head batch
        let lblTensor :: Tensor '[1, 10] = constant (fromIntegral lbl :: HsReal)
        let imageTensor :: Tensor '[1, DataDim] = resizeAs image

        let (loss, (netGrad, xsGrad)) = backprop2 (\net' x' -> Bp.mSECriterion lblTensor $ forward net' x') net imageTensor

        print loss

        go (c+1) (updateNet net (lr, netGrad))

printBatch :: [(Tensor '[DataDim], Int)] -> IO ()
printBatch batch = do
    mapM_ printBatch' batch
    where
        printBatch' :: (Tensor '[DataDim], Int) -> IO ()
        printBatch' (tensor, lbl) = do
            putStrLn "==="
            printf "label: %d\n" lbl
            putStr . unlines $
                [(render . ((P.!!) $ tensordata $ tensor) . (r*28 +)) <$> [0..27] | r <- [0..27]]


main :: IO ()
main = do
    let path = "./mnist"
    let batchSize = 12

    imagesBS <- decompress <$> BS.readFile (printf "%s/%s" path "train-images-idx3-ubyte.gz")
    labelsBS <- decompress <$> BS.readFile (printf "%s/%s" path "train-labels-idx1-ubyte.gz")

    gen <- newRNG
    RNG.manualSeed gen seedVal
    batch <- genBatch gen (imagesBS, labelsBS) batchSize
    printBatch batch

    putStrLn "Done"
