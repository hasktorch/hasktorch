{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Data.Functor
import Control.Monad

import Numeric.Backprop as Bp
import Prelude as P
import Torch.Double as T hiding (add)
import Torch.Double.NN.Linear (Linear(..), linearBatch)
import qualified Torch.Core.Random as RNG

import Text.Printf

import Data.List.Split
import Data.Int

type DataDim = 784
type BatchSize = 64


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

    -- print digit
    putStrLn "========="
    putStr . unlines $
        [(render . ((P.!!) $ tensordata $ tensor) . (r*28 +)) <$> [0..27] | r <- [0..27]]

    pure tensor


genBatch :: RNG.Generator
         -> BS.ByteString
         -> Int
         -> IO ([Tensor '[DataDim]])
genBatch gen images batchSize = do
    imageIdxs <- (fmap $ map $ P.round . (60000 *)) $ replicateM batchSize $ RNG.uniform gen 0 1
    let tensorIOs = [image2tensor imageIdx images | imageIdx <- imageIdxs]
    let tensorsIO = sequence tensorIOs
    tensorsIO


main :: IO ()
main = do
    let path = "./mnist"
    let batchSize = 12

    imagesBS <- decompress <$> BS.readFile (printf "%s/%s" path "train-images-idx3-ubyte.gz")
    labelsBS <- decompress <$> BS.readFile (printf "%s/%s" path "train-labels-idx1-ubyte.gz")

    gen <- newRNG
    RNG.manualSeed gen seedVal
    batch <- genBatch gen imagesBS batchSize

    putStrLn "Done"
