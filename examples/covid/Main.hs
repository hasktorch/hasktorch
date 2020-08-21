{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import CovidData
import CovidUtil
import Data.Csv
import qualified Data.Map as M
import qualified Data.Set as S
import Data.Time
import qualified Data.Vector as V
import GHC.Generics (Generic)
import qualified Graphics.Vega.VegaLite as VL hiding (sample, shape)
import Pipes
import Pipes.Prelude (drain, toListM)
import Text.Pretty.Simple (pPrint)
import TimeSeriesModel
import Torch
import Torch as T

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

main :: IO ()
main = do
  putStrLn "Loading Data"
  dataset <- loadDataset "data/us-counties.csv"
  putStrLn "Preprocessing Data"
  modelData <- prepData dataset
  let tensorData = prepTensors modelData

  let tIndices = asTensor (fipsIdxs modelData)
  let embedDim = 2
  weights <- randnIO' [M.size $ fipsMap modelData, 2]
  let locEmbed = embedding' weights tIndices
  print $ indexSelect' 0 [0 .. 10] locEmbed

  -- define fipsSpace
  let fipsList = M.keys . fipsMap $ modelData
  putStrLn "Number of counties:"
  print $ length fipsList

  plotTS (fipsMap modelData) tensorData "25025"
  plotTS (fipsMap modelData) tensorData "51059"
  plotTS (fipsMap modelData) tensorData "48113"
  plotTS (fipsMap modelData) tensorData "06037"
  plotTS (fipsMap modelData) tensorData "06075"

{-
  let regionData = filterOn tFips (eq (1183) tensorData)
  let t2vd = 6
  let inputDim = 3193 + t2vd + 1 -- # counties + t2vDim + county of interest count
  initializedModel <- initModel 3193 t2vd 12
  -}

  let optimSpec = optimSpec

  -- pPrint $ dataset V.! 1
  putStrLn "Done"
