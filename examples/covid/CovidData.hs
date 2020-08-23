{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module CovidData where

import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Map as M
import qualified Data.Set as S
import Data.Time
import qualified Data.Vector as V
import GHC.Generics (Generic)
import Torch as T

import CovidUtil

-- This is a placeholder for this example until we have a more formal data loader abstraction
--
class Dataset d a where
  getItem ::
    d ->
    Int -> -- index
    Int -> -- batchSize
    a

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

-- instance FromNamedRecord UsCounties

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

instance Dataset TensorData TensorData where
  getItem dataset index batchSize = 
    filterOn tTimes timeFilter2
    . filterOn tTimes timeFilter1 
    $ dataset
    where
      timeFilter1 = \t -> ge t (asTensor (fromIntegral index :: Float))
      timeFilter2 = \t -> lt t (asTensor (fromIntegral (index + batchSize) :: Float))

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
diff :: Tensor -> Tensor
diff t = (indexSelect' 0 [1 .. len -1] t) - (indexSelect' 0 [0 .. len -2] t)
  where
    len = shape t !! 0

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
-- Clamping is done since data artifacts can cause total changes to go negative - see https://github.com/nytimes/covid-19-data/issues/425
plotTS fips2idx tensorData fips = do
  let fipsIdx = fips2idx M.! fips
  let newCases =
        trim
          . clampMin 0.0
          . diff
          . tCases
          . filterOn tFips (eq $ asTensor fipsIdx)
          $ tensorData
  tensorSparkline newCases
