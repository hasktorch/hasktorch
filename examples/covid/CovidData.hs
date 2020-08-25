{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module CovidData where

import CovidUtil
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Map as M
import Data.Maybe (fromJust)
import qualified Data.Set as S
import Data.Time
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Safe (atMay)
import Safe.Exact (dropExactMay, takeExactMay)
import Torch as T
import Prelude as P

-- | This is a placeholder for this example until we have a more formal data loader abstraction
class Dataset d a | d -> a where
  getItem ::
    d -> -- dataset
    Int -> -- index
    Int -> -- batchSize
    a -- a container type for dataset itemsitem type

-- | Single record used to parse CSV
data UsCounties = UsCounties
  { date :: String,
    county :: String,
    state :: String,
    fips :: String,
    cases :: Int,
    deaths :: Int
  }
  deriving (Eq, Generic, Show)

instance FromRecord UsCounties

-- | Records aggregated into lists + fips -> ID mapping
data ModelData = ModelData
  { timePoints :: [Int],
    fipsStrs :: [String],
    fipsIdxs :: [Int],
    fipsMap :: M.Map String Int,
    caseCounts :: [Int],
    deathCounts :: [Int]
  }
  deriving (Eq, Generic, Show)

-- | Records packaged into tensors
data TensorData = TensorData
  { tTimes :: Tensor,
    tFips :: Tensor,
    tCases :: Tensor,
    tDeaths :: Tensor
  }
  deriving (Eq, Generic, Show)

instance Dataset TensorData TensorData where
  getItem dataset index batchSize =
    filterOn tTimes timeFilter2
      . filterOn tTimes timeFilter1
      $ dataset
    where
      timeFilter1 = \t -> ge t (asTensor (fromIntegral index :: Float))
      timeFilter2 = \t -> lt t (asTensor (fromIntegral (index + batchSize) :: Float))

-- | Representation of a 1D time series as observations and lists of time points
-- 1st list dimension = observation #
-- 2nd list dimension = time point
-- Tensor = 1x1 value
data TimeSeriesData = TimeSeriesData
  { pastObs :: Series, -- all observations before split point
    nextObs :: Series -- next window
  }
  deriving (Show)

newtype Series = Series [[Tensor]] deriving (Show)

newtype Obs = Obs Int deriving (Show)

newtype Time = Time Int deriving (Show)

-- | Get an observation window
getObsBatch :: Obs -> Obs -> Series -> Maybe Series
getObsBatch (Obs obsIdx) (Obs obsNum) (Series series) =
  pure series >>= dropExactMay obsIdx >>= takeExactMay obsNum >>= pure . Series

-- | Get an observation window
getObs :: Obs -> Series -> Maybe [Tensor]
getObs (Obs obsIdx) (Series series) = atMay series obsIdx

-- | Get an observation window, unsafe convenience function
getObs' :: Int -> Series -> [Tensor]
getObs' obs = fromJust . getObs (Obs obs)

-- | Get a time point from an observation window
getTime :: Obs -> Time -> Series -> Maybe Tensor
getTime (Obs obsIdx) (Time timeIdx) (Series series) =
  atMay series obsIdx >>= atMay' timeIdx >>= pure
  where
    atMay' = flip atMay

-- | Get a time point from an observation window, unsafe convenience function
getTime' :: Int -> Int -> Series -> Tensor
getTime' obsIdx timeIdx (Series series) = series !! obsIdx !! timeIdx

instance Dataset TimeSeriesData (Series, Series) where
  getItem TimeSeriesData {..} index batchSize = (slice pastObs, slice nextObs)
    where
      slice series = fromJust $ getObsBatch (Obs index) (Obs batchSize) series

-- | Parse a CSV file of county level data
loadDataset :: String -> IO (V.Vector UsCounties)
loadDataset fileName = do
  csvData <- BL.readFile fileName
  case decode HasHeader csvData of
    Left err -> error err
    Right (v :: V.Vector UsCounties) -> pure v

-- | Transform parsed data into struct-of-lists shape useful for modeling
prepData :: V.Vector UsCounties -> IO ModelData
prepData dataset = do
  let fipsSet = S.fromList . V.toList $ fips <$> dataset
  let idxMap = M.fromList $ zip (S.toList fipsSet) [0 .. length fipsSet - 1]
  let indices = V.toList $ ((M.!) idxMap) <$> (fips <$> dataset)
  times <- datesToTimepoints "2020-01-21" (V.toList $ date <$> dataset)
  pure
    ModelData
      { timePoints = times,
        fipsStrs = V.toList $ fips <$> dataset,
        fipsIdxs = indices,
        fipsMap = idxMap,
        caseCounts = V.toList $ cases <$> dataset,
        deathCounts = V.toList $ deaths <$> dataset
      }

-- | Transform struct-of-lists shape to tensor values
prepTensors :: ModelData -> TensorData
prepTensors modelData =
  TensorData
    { tTimes = (asTensor . timePoints $ modelData),
      tFips = (asTensor . fipsIdxs $ modelData),
      tCases = (asTensor . caseCounts $ modelData),
      tDeaths = (asTensor . deathCounts $ modelData)
    }

-- | Treat TensorData as a pseudo-dataframe, filter the field specified by `getter` using
-- the predicate function
filterOn ::
  (TensorData -> Tensor) -> -- getter
  (Tensor -> Tensor) -> -- predicate
  TensorData -> -- input data
  TensorData -- filtered data
filterOn getter pred tData =
  TensorData
    { tTimes = selector (tTimes tData),
      tFips = selector (tFips tData),
      tCases = selector (tCases tData),
      tDeaths = selector (tDeaths tData)
    }
  where
    selector :: Tensor -> Tensor
    selector = indexSelect 0 (squeezeAll . nonzero . pred . getter $ tData)

-- | Convert date strings to days since 1/21/2020 (the first date of this dataset)
datesToTimepoints :: String -> [String] -> IO [Int]
datesToTimepoints day0 dateStrings = do
  firstDay :: Day <- parseTimeM False defaultTimeLocale "%F" day0
  days :: [Day] <- sequence $ parseTimeM False defaultTimeLocale "%F" <$> dateStrings
  pure $ fromIntegral <$> flip diffDays firstDay <$> days

-- | calculate first differences of a tensor
firstDiff :: Tensor -> Tensor
firstDiff t = (indexSelect' 0 [1 .. len -1] t) - (indexSelect' 0 [0 .. len -2] t)
  where
    len = shape t !! 0

-- | Transform total cases tensor to new case counts, clamp at 0 for data abnormalities
newCases :: Tensor -> Tensor
newCases totalCases = clampMin 0.0 . firstDiff $ totalCases

-- | trim leading zeros from a tensor
trim :: Tensor -> Tensor
trim t =
  if hasVal
    then
      let firstNonzero = asValue $ nz ! (0 :: Int)
       in indexSelect' 0 [firstNonzero .. len -1] t
    else t
  where
    len = shape t !! 0
    nz = squeezeAll . nonzero $ t
    hasVal = T.all $ toDType Bool nz

-- | Plot time series
-- Clamping is done since data artifacts can cause total changes to go negative 
-- see https://github.com/nytimes/covid-19-data/issues/425
plotTS fips2idx tensorData fips = do
  let fipsIdx = fips2idx M.! fips
  let newCases =
        trim
          . clampMin 0.0
          . firstDiff
          . tCases
          . filterOn tFips (eq $ asTensor fipsIdx)
          $ tensorData
  tensorSparkline newCases

{- TimeSeries Type Operations -}

-- | Given a 1D time series tensor
-- create lists of tensors from time point 1..t
-- and prospective windows of a fixed size
expandToSplits ::
  Int -> -- Size of future window
  Tensor ->
  TimeSeriesData -- Example Index, Time Index, 1x1 Tensor
expandToSplits futureWindow timeseries =
  TimeSeriesData
    (Series [P.take i casesList | i <- [1 .. length casesList - futureWindow]])
    ( Series
        [ P.take futureWindow (P.drop i casesList)
          | i <- [1 .. length casesList - futureWindow]
        ]
    )
  where
    casesList =
      reshape [1, 1]
        . asTensor
        <$> (fromIntegral <$> (asValue timeseries :: [Int]) :: [Float])

-- | Given a 1D time series tensor
-- create lists of tensors from time point 1..t
-- and prospective windows for the remainder of the series
expandToSplitsAll ::
  Tensor ->
  TimeSeriesData -- Example Index, Time Index, 1x1 Tensor
expandToSplitsAll timeseries =
  TimeSeriesData
    (Series [P.take i casesList | i <- [1 .. length casesList - 1]])
    (Series [P.drop i casesList | i <- [1 .. length casesList - 1]])
  where
    casesList =
      reshape [1, 1]
        . asTensor
        <$> (fromIntegral <$> (asValue timeseries :: [Int]) :: [Float])
