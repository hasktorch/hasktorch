{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import AlexNet
    
import GHC.Generics
import Control.Monad (when)
import Text.Regex.TDFA
import System.Directory
import System.Random
import Data.List
import qualified Data.Map.Strict as S

import Torch.Optim
import Torch.Autograd
import Torch.Serialize
import Torch.Vision
import Torch.NN
import Torch.Functional
import Torch.TensorFactories
import Torch.TensorOptions
import Torch.Tensor
import qualified Torch.DType as D
import qualified Torch.Functional.Internal as I

data DataSet = DataSet {
    images  :: [Tensor],
    labels  :: [Tensor],
    classes :: [String]
} deriving (Show)

shuffle :: [a] -> IO[a]
shuffle x = if length x < 2 then return x else do
    i <- System.Random.randomRIO (0, length(x)-1)
    r <- shuffle (take i x ++ drop (i+1) x)
    return (x!!i : r)

splitList :: [FilePath] -> Float -> IO([FilePath], [FilePath])
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
createLabels = ( =~ ("[A-Z]*[a-z]*" :: String))

normalize :: Tensor -> Tensor
normalize img = I.unsqueeze (cat (Dim 0) [r', g', b']) 0
    where
        img' = divScalar (255.0::Float) img
        [r, g, b] = I.split (reshape [3, 224, 224] img') 1 0 
        r' = divScalar (0.229::Float) (subScalar(0.485::Float) r) 
        g' = divScalar (0.224::Float) (subScalar(0.456::Float) g) 
        b' = divScalar (0.225::Float) (subScalar(0.406::Float) b) 

extract' :: Maybe a -> a
extract' (Just x) = x          
extract' Nothing  = undefined  

labelToTensor :: [String] -> [String] -> [Int]
labelToTensor unique  = map (extract' . (m S.!?))
    where m = S.fromList $ zip unique [0..(length unique - 1)]

-- takes in a D.Float chw image and return images of dimension 224 X 224
resize :: Tensor -> Tensor
resize img = reshape [3, 224, 224] $ I.upsample_bilinear2d img (224, 224) True

createDataSet :: [FilePath]  -> IO DataSet
createDataSet directorylist = do
    readList <- mapM (readImageAsRGB8 . ("alexNet/images/" ++ )) directorylist -- readList list of tensors
    let listIL = extractImagesNLabels directorylist readList
        labelList = map (createLabels . fst) listIL 
        imageList = map (toDType D.Float . hwc2chw . snd) listIL
        unique = nub labelList
        batchSize = 16 :: Int
        nImages = length labelList
        nBatches = nImages `div` batchSize
        
        processedImages = I.split (cat (Dim 0) $ map (normalize . resize) imageList) batchSize 0
        processedLabels = I.split (asTensor $ labelToTensor unique labelList) batchSize 0
        dataset = DataSet processedImages processedLabels unique
    return dataset

train :: DataSet -> AlexNetBB -> IO Linear
train trainDataset pretrainedbb = do
    let uniqueClasses = classes trainDataset
    initial <- sample $ LinearSpec 4096 $ length uniqueClasses
    trained <- foldLoop initial 5000 $
        \state iter -> do
            k <- randomRIO (1, length $ images trainDataset)
            let imageBatch = (images trainDataset) !! (k - 1) 
                labelBatch = (labels trainDataset) !! (k - 1) 
                loss = nllLoss' labelBatch $ logSoftmax (Dim 1) $ linear state $ alexnetBBForward pretrainedbb imageBatch
            when (iter `mod` 100 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss for current mini-batch: " ++ show loss
            (newParam, _) <- runStep state GD loss 1e-2
            pure $ replaceParameters state newParam
    return trained

evaluate :: [Tensor] -> AlexNetBB -> Linear -> [Tensor]
evaluate images pretrainedbb trainedFL = map ((argmax (Dim (-1)) KeepDim) . (softmax (Dim (-1)))) unNormalizedScores
    where
        unNormalizedScores =  map (linear trainedFL . (alexnetBBForward pretrainedbb)) images

calcAccuracy :: [Tensor] -> [Tensor] -> Float
calcAccuracy l m = (fromIntegral( toInt (sum $ map (\t -> sumAll $ fst t ==. (flattenAll $ snd t)) $ zip l m)) :: Float) / (fromIntegral((sum $ map (head . shape) m)) :: Float)

main = do
    pretrainedANParams <- load "alexNet/alexNet.pt"
    alexnetBB <- sample alexnetBackBoneSpec
    
    let bbparams = map IndependentTensor $ init $ init pretrainedANParams
        pretrainedbb = replaceParameters alexnetBB bbparams
    
    directoryList <- getDirectoryContents "alexNet/images"
    (trainList, testList) <- splitList (drop 2 directoryList) 0.9
    
    trainDataSet <- createDataSet trainList
    testDataSet <- createDataSet testList
    
    trainedFinalLayer <- train trainDataSet pretrainedbb

    print $ "Accuracy on train-set: " ++ ( show $ calcAccuracy (labels trainDataSet) $ evaluate (images trainDataSet) pretrainedbb trainedFinalLayer)
    print $ "Accuracy on test-set: " ++ ( show $ calcAccuracy (labels testDataSet) $ evaluate (images testDataSet) pretrainedbb trainedFinalLayer)
    
    putStrLn "Done"
