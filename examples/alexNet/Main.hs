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
import qualified GHC.List as G

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
createLabels string = firstMatch
    where (firstMatch, _, _) = (string =~ "_[0-9]+.jpg" :: (String, String, String))

-- normalize :: Tensor -> Tensor
-- normalize t = t `sub` (uns mean) `Torch.Functional.div` uns stddev 
--     where 
--         mean = asTensor ([0.485, 0.456, 0.406] :: [Float])
--         stddev = asTensor ([0.229, 0.224, 0.225] :: [Float])
--         uns t = unsqueeze (Dim 2) $ unsqueeze (Dim 1) t

normalize :: Tensor -> Tensor
normalize img = unsqueeze (Dim 0) (cat (Dim 0) [r', g', b'])
    where
        img' = divScalar (255.0::Float) img
        [r, g, b] = Torch.Functional.split 1 (Dim 0) (reshape [3, 224, 224] img')
        r' = divScalar (0.229::Float) (subScalar (0.485::Float) r) 
        g' = divScalar (0.224::Float) (subScalar (0.456::Float) g) 
        b' = divScalar (0.225::Float) (subScalar (0.406::Float) b) 

extract' :: Maybe a -> a
extract' (Just x) = x          
extract' Nothing  = undefined  

labelToTensor :: [String] -> [String] -> [Int]
labelToTensor unique  = map (extract' . (m S.!?))
    where m = S.fromList $ zip unique [0..(length unique - 1)]

createDataSet :: [FilePath]  -> IO DataSet
createDataSet directorylist = do
    readList <- mapM (readImageAsRGB8 . ("alexNet/images/" ++ )) directorylist
    let listIL = extractImagesNLabels directorylist readList
        labelList = map (createLabels . fst) listIL 
        imageList = map (toDType D.Float . hwc2chw . snd) listIL
        unique = nub labelList
        batchSize = 16 :: Int
        rszdImgs =  map (upsampleBilinear2d (224, 224) True) imageList
        nrmlzdImgs = map normalize rszdImgs
        prcssdLbls = labelToTensor unique labelList
        
        images = Torch.Functional.split  batchSize (Dim 0) (cat (Dim 0) nrmlzdImgs)
        labels = Torch.Functional.split  batchSize (Dim 0) (asTensor prcssdLbls)
        dataset = DataSet images labels unique
    print $ shape $ head rszdImgs
    return dataset

-- dataset with unique occurence for each class 
writeDataset :: DataSet -> IO()
writeDataset ds = do
    let fnames = map (++".bmp") $ map (\ind -> (classes ds) !! ind) $ G.concat $ map (asValue) $ G.concat $ map (Torch.Functional.split 1 (Dim 0)) $ labels ds
        imgs = G.concat $ map (Torch.Functional.split 1 (Dim 0) ) $ images ds
    mapM_ (\(f, i) -> writeBitmap f $ toDType D.UInt8 $  chw2hwc i) $ zip fnames imgs

calcLoss :: [Tensor] -> [Tensor] -> [Tensor]
calcLoss targets predictions = map (\(t, p)-> nllLoss' t p) $ zip targets predictions

oneEpoch :: DataSet -> AlexNetBB -> Linear -> Adam -> IO (Linear, Adam)
oneEpoch dataset pretrainedbb model optim = do
    (newEpochModel, newEpochOptim) <- foldLoop (model, optim) (length $ labels dataset) $
        \(iterModel, iterOptim) iter -> do
            let imageBatch = (images dataset) !! (iter - 1) 
                labelBatch = (labels dataset) !! (iter - 1) 
                scores = logSoftmax (Dim 1) $ linear iterModel $ alexnetBBForward pretrainedbb imageBatch
                loss = nllLoss' labelBatch scores
            when (iter `mod` 30 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss for current mini-batch: " ++ show loss
            (newParam, newOptim) <- runStep iterModel iterOptim loss 1e-4
            pure (replaceParameters iterModel newParam, newOptim)
    return (newEpochModel, newEpochOptim)

train :: DataSet -> DataSet -> AlexNetBB -> IO Linear
train trainDataset valDataset pretrainedbb = do
    let uniqueClasses = classes trainDataset
    initModel <- sample $ LinearSpec 4096 $ length uniqueClasses
    let initOptim = mkAdam 0 0.9 0.999 (flattenParameters initModel) 
    (trained, _) <- foldLoop (initModel, initOptim) 5 $
        \(state, optim) iter -> do
            print $ calcLoss (labels valDataset) $ map ((logSoftmax (Dim 1)) . (linear state) . (alexnetBBForward pretrainedbb)) $ images valDataset
            oneEpoch trainDataset pretrainedbb state optim     
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
    (trainList, testList) <- splitList (filter (\x-> if x=="." || x==".." then False else True) directoryList) 0.9
    
    trainDataSet <- createDataSet trainList
    testDataSet <- createDataSet testList
    
    trainedFinalLayer <- train trainDataSet testDataSet pretrainedbb

    print $ "Accuracy on train-set: " ++ ( show $ calcAccuracy (labels trainDataSet) $ evaluate (images trainDataSet) pretrainedbb trainedFinalLayer)
    print $ "Accuracy on test-set: " ++ ( show $ calcAccuracy (labels testDataSet) $ evaluate (images testDataSet) pretrainedbb trainedFinalLayer)
    
    putStrLn "Done"
