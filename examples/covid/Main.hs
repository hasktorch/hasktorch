{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Control.Arrow (Arrow (first))
import Control.Monad ((>=>), foldM, when)
import Control.Monad.Cont (ContT (ContT), runCont, runContT)
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import Data.List (nub)
import qualified Data.Map as M
import qualified Data.Set as S
import Data.Time
import Data.Time.Calendar.OrdinalDate (toOrdinalDate)
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Graphics.Vega.VegaLite hiding (sample, shape)
import Pipes
import Pipes.Prelude (drain, toListM)
import Text.Pretty.Simple (pPrint)
import Torch
import Torch as T
import Torch.Data.CsvDataset
import Torch.Data.Pipeline (FoldM (FoldM))
-- import Torch.Data.StreamedPipeline (MonadBaseControl, pmap, makeListT)
import Torch.Data.StreamedPipeline

data UsCounties = UsCounties
  { date :: String,
    county :: String,
    state :: String,
    fips :: String,
    cases :: Int,
    deaths :: Int
  }
  deriving (Eq, Generic, Show)


data ModelData = ModelData
  { timePoints :: [Int],
    fipsStrs :: [String],
    fipsIdxs :: [Int],
    fipsMap :: M.Map String Int,
    caseCounts :: [Int],
    deathCounts :: [Int]
  }
  deriving (Eq, Generic, Show)

data TensorData = TensorData
  { tTimes :: Tensor,
    tFips :: Tensor,
    tCases :: Tensor,
    tDeaths :: Tensor
  }
  deriving (Eq, Generic, Show)

instance FromRecord UsCounties

-- instance FromNamedRecord UsCounties

modelTrain = undefined

-- TODO - use this
-- counties = (csvDataset @UsCounties "data/us-counties.csv") {batchSize = 4, shuffle = Nothing}

loadDataset fileName = do
  csvData <- BL.readFile fileName
  case decode HasHeader csvData of
    Left err -> error err
    Right (v :: V.Vector UsCounties) -> pure v


initializeEmbedding :: Int -> Tensor -> IO Tensor
initializeEmbedding embedDim t =
  randIO' [nUniq, embedDim]
  where 
    (uniqT, _, _) = (T.uniqueDim 0 True False False t)
    nUniq = shape uniqT !! 0

prepData :: V.Vector UsCounties -> IO ModelData
prepData dataset = do
  let fipsSet = S.fromList . V.toList $ fips <$> dataset
  let idxMap = M.fromList $ zip (S.toList fipsSet)  [0 .. length fipsSet - 1] 
  let indices = V.toList $ ((M.!) idxMap) <$> (fips <$> dataset)
  times <- datesToTimepoints (V.toList $ date <$> dataset)
  pure ModelData {
    timePoints=times,
    fipsStrs=V.toList $ fips <$> dataset,
    fipsIdxs=indices,
    fipsMap=idxMap,
    caseCounts=V.toList $ cases <$> dataset,
    deathCounts=V.toList $ deaths <$> dataset
  }

prepTensors :: ModelData -> TensorData
prepTensors modelData =
  TensorData {
    tTimes = (asTensor . timePoints $ modelData),
    tFips = (asTensor . fipsIdxs $ modelData),
    tCases = (asTensor . caseCounts $ modelData),
    tDeaths = (asTensor . deathCounts $ modelData)
    }

filterOn 
  :: (TensorData -> Tensor) -- getter
  -> (Tensor -> Tensor) -- predicate
  -> TensorData -- input data
  -> TensorData -- filtered data
filterOn getter pred tData =
  TensorData {
    tTimes = selector (tTimes tData),
    tFips = selector (tFips tData),
    tCases = selector (tCases tData),
    tDeaths = selector (tDeaths tData)
    }
  where
    selector :: Tensor -> Tensor
    selector = indexSelect 0 (squeezeAll . nonzero . pred . getter $ tData)
  

datesToTimepoints :: [String] -> IO [Int]
datesToTimepoints dateStrings = do
  firstDay :: Day <- parseTimeM False defaultTimeLocale "%F" "2020-01-21"
  days :: [Day] <- sequence $ parseTimeM False defaultTimeLocale "%F" <$> dateStrings
  pure $ fromIntegral <$> flip diffDays firstDay <$> days

main :: IO ()
main = do
  -- let counties = (csvDataset @UsCounties "data/us-counties.csv") { batchSize = 4 , shuffle = Nothing}
  -- (countiesList) <- makeListT counties (Select $ yield ())

  putStrLn "Loading Data"
  dataset <- loadDataset "data/us-counties.csv"
  putStrLn "Preprocessing Data"
  modelData <- prepData dataset
  let tensorData = prepTensors modelData

  let tIndices = asTensor (fipsIdxs modelData)
  let embedDim = 2
  weights <- randnIO' [M.size $ fipsMap modelData, 2]
  let locEmbed = embedding' weights tIndices
  print $ indexSelect' 0 [0..10] locEmbed

  -- Autoencoder define fipsSpace
  let fipsList = M.keys . fipsMap $ modelData

  print $ length $ Prelude.filter (\x -> x ==2102) (fipsIdxs modelData)
  print $ length $ Prelude.filter (\x -> x == 739) (fipsIdxs modelData)

  pPrint $ filterOn tTimes (eq $ asTensor (20 :: Int)) tensorData 
  pPrint $ filterOn tFips (eq $ asTensor (1 :: Int)) tensorData


  -- pPrint $ dataset V.! 1
  putStrLn "Done"
