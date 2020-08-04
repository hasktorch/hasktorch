{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Control.Arrow (Arrow (first))
import Control.Monad ((>=>), foldM, when)
import Control.Monad.Cont (ContT (ContT), runCont, runContT)
import qualified Data.ByteString.Lazy as BL

import Data.Csv
import qualified Data.Map as M
import qualified Data.Set as S
import Data.Time
import Data.Time.Calendar.OrdinalDate (toOrdinalDate)
import qualified Data.Vector as V
import GHC.Generics (Generic)
import qualified Graphics.Vega.VegaLite as VL hiding (sample, shape)
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

-- | Convert a series into a sparkline string (from clisparkline library)
series2sparkline :: RealFrac a => [a] -> String
series2sparkline vs =
  let maxv = if null vs then 0 else maximum vs
  in map (num2sparkchar maxv) vs
  where
    sparkchars = "_▁▂▃▄▅▆▇█" 
    num2sparkchar maxv curv =
      sparkchars !!
        (Prelude.floor $ (curv / maxv) * (fromIntegral (length sparkchars - 1)))
  

tensorSparkline :: Tensor -> IO ()
tensorSparkline t = putStrLn $ (series2sparkline (asValue t' :: [Float])) ++ (" | Max: " ++ show (asValue maxValue :: Float))
  where 
    t' = toDType Float t
    maxValue = T.max t'

diff t = (indexSelect' 0 [1..len-1] t) - (indexSelect' 0 [0..len-2] t)
  where len = shape t !! 0

trim t = 
  if hasVal then
    let firstNonzero = asValue $ nz ! (0 :: Int)
      in indexSelect' 0 [firstNonzero..len-1] t
  else t
  where 
    len = shape t !! 0
    nz = squeezeAll . nonzero $ t 
    hasVal = T.all $ toDType Bool nz

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

  print $ length $ Prelude.filter (\x -> x == 2102) (fipsIdxs modelData)
  print $ length $ Prelude.filter (\x -> x == 739) (fipsIdxs modelData)

  pPrint $ filterOn tTimes (eq $ asTensor (20 :: Int)) tensorData 
  pPrint $ filterOn tFips (eq $ asTensor (1 :: Int)) tensorData

  -- index 1222 = FIPS 25025
  let newCases = trim
        . clampMin 0.0 
        . diff 
        . tCases 
        . filterOn tTimes ((flip lt) $ asTensor (120 :: Int))
        . filterOn tFips (eq $ asTensor (1222 :: Int)) $ tensorData 

  tensorSparkline newCases
  -- clamping - see https://github.com/nytimes/covid-19-data/issues/425


  -- pPrint $ dataset V.! 1
  putStrLn "Done"
