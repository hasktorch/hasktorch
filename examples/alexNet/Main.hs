{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import AlexNet (AlexNetBB, alexnetBBForward, alexnetBackBoneSpec)
import Control.Monad (when)
import Data.List (nub)
import qualified Data.Map.Strict as S
import qualified GHC.List as G
import System.Directory (getDirectoryContents)
import System.Random (randomRIO)
import Text.Regex.TDFA ((=~))
import Torch.Autograd (IndependentTensor (..))
import qualified Torch.DType as D
import Torch.Functional as F hiding (take)
import Torch.NN as N
import Torch.Optim (Adam, foldLoop, mkAdam, runStep)
import Torch.Serialize (load)
import Torch.Tensor (Tensor, asTensor, reshape, shape, toInt)
import qualified Torch.Typed.Vision as V hiding (getImages')
import qualified Torch.Vision as V

data DataSet = DataSet
  { images :: [Tensor],
    labels :: [Tensor],
    classes :: [String]
  }
  deriving (Show)

shuffle :: [a] -> IO [a]
shuffle x =
  if length x < 2
    then return x
    else do
      i <- randomRIO (0, length (x) -1)
      r <- shuffle (take i x ++ drop (i + 1) x)
      return (x !! i : r)

splitList :: [FilePath] -> Float -> IO ([FilePath], [FilePath])
splitList list ratio = do
  sList <- shuffle list
  return (take splitSize sList, drop splitSize sList)
  where
    splitSize = round $ fromIntegral (length list) * ratio

extractImagesNLabels :: [FilePath] -> [Either String Tensor] -> [(FilePath, Tensor)]
extractImagesNLabels [] [] = []
extractImagesNLabels (p : ps) (Left err : ts) = extractImagesNLabels ps ts
extractImagesNLabels (p : ps) (Right t : ts) = (p, t) : extractImagesNLabels ps ts

createLabels :: FilePath -> String
createLabels string = firstMatch
  where
    (firstMatch, _, _) = (string =~ "_[0-9]+.jpg" :: (String, String, String))

-- normalizeT :: Tensor -> Tensor
-- normalizeT t = (t `F.sub` (uns mean)) `F.div` (uns stddev)
--     where
--         mean = asTensor ([0.485, 0.456, 0.406] :: [Float])
--         stddev = asTensor ([0.229, 0.224, 0.225] :: [Float])
--         uns t = F.unsqueeze (F.Dim 2) $ F.unsqueeze (F.Dim 1) t

normalize :: Tensor -> Tensor
normalize img = F.unsqueeze (F.Dim 0) (F.cat (F.Dim 0) [r', g', b'])
  where
    img' = F.divScalar (255.0 :: Float) img
    [r, g, b] = F.split 1 (F.Dim 0) (reshape [3, 224, 224] img')
    r' = F.divScalar (0.229 :: Float) (F.subScalar (0.485 :: Float) r)
    g' = F.divScalar (0.224 :: Float) (F.subScalar (0.456 :: Float) g)
    b' = F.divScalar (0.225 :: Float) (F.subScalar (0.406 :: Float) b)

extract' :: Maybe a -> a
extract' (Just x) = x
extract' Nothing = undefined

labelToTensor :: [String] -> [String] -> [Int]
labelToTensor unique = map (extract' . (m S.!?))
  where
    m = S.fromList $ zip unique [0 .. (length unique - 1)]

createDataSet :: [FilePath] -> [String] -> IO DataSet
createDataSet directorylist classes = do
  readList <- mapM (V.readImageAsRGB8 . ("alexNet/images/" ++)) directorylist
  let listIL = extractImagesNLabels directorylist readList
      labelList = map (createLabels . fst) listIL
      imageList = map (F.toDType D.Float . V.hwc2chw . snd) listIL
      batchSize = 16 :: Int
      rszdImgs = map (F.upsampleBilinear2d (224, 224) True) imageList
      nrmlzdImgs = map normalize rszdImgs
      prcssdLbls = labelToTensor classes labelList

      images = F.split batchSize (F.Dim 0) $ F.cat (F.Dim 0) nrmlzdImgs
      labels = F.split batchSize (F.Dim 0) (asTensor prcssdLbls)
      dataset = DataSet images labels classes
  return dataset

fromMNIST :: V.MnistData -> DataSet
fromMNIST ds =
  let nImages = V.length ds `Prelude.div` 10
      batchSize = 16 :: Int
      labelT = V.getLabels' nImages ds [0 .. (nImages -1)]
      imagesT = V.getImages' nImages 784 ds [0 .. (nImages -1)]
      rimages = F.repeatInterleaveScalar (reshape [nImages, 1, 28, 28] imagesT) 3 1
      rszdImgs = F.upsampleBilinear2d (224, 224) True rimages
      nrmlzdImgs = map normalize $ F.split 1 (F.Dim 0) rszdImgs

      images = F.split batchSize (F.Dim 0) (F.cat (F.Dim 0) nrmlzdImgs)
      labels = F.split batchSize (F.Dim 0) labelT
      dataset = DataSet images labels $ map show [0 .. 9]
   in dataset

calcLoss :: [Tensor] -> [Tensor] -> [Tensor]
calcLoss targets predictions = take 10 $ map (\(t, p) -> F.nllLoss' t p) $ zip targets predictions

oneEpoch :: DataSet -> AlexNetBB -> N.Linear -> Adam -> IO (N.Linear, Adam)
oneEpoch dataset pretrainedbb model optim = do
  (newEpochModel, newEpochOptim) <- foldLoop (model, optim) (length $ labels dataset) $
    \(iterModel, iterOptim) iter -> do
      let imageBatch = (images dataset) !! (iter - 1)
          labelBatch = (labels dataset) !! (iter - 1)
          scores = F.logSoftmax (F.Dim 1) $ N.linear iterModel $ alexnetBBForward pretrainedbb imageBatch
          loss = F.nllLoss' labelBatch scores
      when (iter `mod` 30 == 0) $ do
        putStrLn $ "Iteration: " ++ show iter ++ " | Loss for current mini-batch: " ++ show loss
      runStep iterModel iterOptim loss 1e-4
  return (newEpochModel, newEpochOptim)

train :: DataSet -> DataSet -> AlexNetBB -> IO N.Linear
train trainDataset valDataset pretrainedbb = do
  let uniqueClasses = classes trainDataset
  initModel <- sample $ N.LinearSpec 4096 $ length uniqueClasses
  let initOptim = mkAdam 0 0.9 0.999 (N.flattenParameters initModel)
  (trained, _) <- foldLoop (initModel, initOptim) 5 $
    \(state, optim) iter -> do
      print $ calcLoss (labels valDataset) $ map ((F.logSoftmax (F.Dim 1)) . (N.linear state) . (alexnetBBForward pretrainedbb)) $ images valDataset
      oneEpoch trainDataset pretrainedbb state optim
  return trained

evaluate :: Tensor -> AlexNetBB -> N.Linear -> Tensor
evaluate images pretrainedbb trainedFL = F.argmax (F.Dim (-1)) F.KeepDim $ F.softmax (F.Dim (-1)) unNormalizedScores
  where
    unNormalizedScores = N.linear trainedFL $ alexnetBBForward pretrainedbb images

calcAccuracy :: (Tensor, Tensor) -> AlexNetBB -> N.Linear -> Float
calcAccuracy (targets, inputs) pretrainedbb trainedFL = correctPreds / totalPreds
  where
    correctPreds = fromIntegral (toInt $ F.sumAll $ F.eq (F.flattenAll predictions) targets) :: Float
    totalPreds = fromIntegral (head $ shape targets) :: Float
    predictions = evaluate inputs pretrainedbb trainedFL

main = do
  pretrainedANParams <- load "alexNet/alexNet.pt"
  alexnetBB <- sample alexnetBackBoneSpec

  let bbparams = map IndependentTensor $ init $ init pretrainedANParams
      pretrainedbb = N.replaceParameters alexnetBB bbparams

  directoryList <- getDirectoryContents "alexNet/images"
  let filteredDirectoryList = filter (\x -> if x == "." || x == ".." then False else True) directoryList
  (trainList, testList) <- splitList filteredDirectoryList 0.9

  let classes = nub $ map createLabels filteredDirectoryList
  trainDataSet <- createDataSet trainList classes
  testDataSet <- createDataSet testList classes

  -- (trainData, testData) <- V.initMnist "datasets/mnist"
  -- trainDataSet <- fromMNIST trainData
  -- testDataSet <- fromMNIST testData

  trainedFinalLayer <- train trainDataSet testDataSet pretrainedbb

  print $ "Accuray on train-set:" ++ show ((foldl (\acc t -> acc + (calcAccuracy t pretrainedbb trainedFinalLayer)) 0 $ zip (labels trainDataSet) (images trainDataSet)) / (fromIntegral $ length $ labels trainDataSet :: Float))
  print $ "Accuray on test-set:" ++ show ((foldl (\acc t -> acc + (calcAccuracy t pretrainedbb trainedFinalLayer)) 0 $ zip (labels testDataSet) (images testDataSet)) / (fromIntegral $ length $ labels testDataSet :: Float))

  putStrLn "Done"
