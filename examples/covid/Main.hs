{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import CovidData
import Data.Csv
import qualified Data.Map as M
import qualified Data.Set as S
import Data.Text.Lazy (unpack)
import Data.Time
import qualified Data.Vector as V
import GHC.Generics (Generic)
import qualified Graphics.Vega.VegaLite as VL hiding (sample, shape)
import Pipes
import Pipes.Prelude (drain, toListM)
import Text.Pretty.Simple (pPrint, pShow)
import TimeSeriesModel
import Torch
import Torch as T
import Torch.NN.Recurrent.Cell.LSTM

plotExampleData modelData tensorData = do
  plotTS (fipsMap modelData) tensorData "25025"
  plotTS (fipsMap modelData) tensorData "51059"
  plotTS (fipsMap modelData) tensorData "48113"
  plotTS (fipsMap modelData) tensorData "06037"
  plotTS (fipsMap modelData) tensorData "06075"

-- | Transform total cases tensor to new case counts, clamp at 0 for data abnormalities
newCases totalCases = clampMin 0.0 . diff $ totalCases

data TimeSeries = TimeSeries
  { -- 1st list dimension = observation #
    -- 2nd  list dimension = time point
    -- Tensor = 1x1 value
    pastObs :: [[Tensor]], -- all observations before split point
    futureObs :: [[Tensor]], -- all observations after split point
    nextObs :: [[Tensor]] -- next window
  }
  deriving (Eq)

instance Show TimeSeries where
  show = unpack . pShow

-- | Given a 1D time series tensor, create lists of tensors from time point 1..t
expandToSplits ::
  Int -> -- Size of future window
  Tensor ->
  TimeSeries -- Example Index, Time Index, 1x1 Tensor
expandToSplits futureWindow timeseries =
  TimeSeries
    [Prelude.take i casesList | i <- [1 .. length casesList - futureWindow]]
    [Prelude.drop i casesList | i <- [1 .. length casesList - futureWindow]]
    [ Prelude.take futureWindow (Prelude.drop i casesList)
      | i <- [1 .. length casesList - futureWindow]
    ]
  where
    casesList =
      reshape [1, 1]
        . asTensor
        <$> (fromIntegral <$> (asValue timeseries :: [Int]) :: [Float])

main :: IO ()
main = do
  putStrLn "Loading Data"
  dataset <- loadDataset "data/us-counties.csv"
  putStrLn "Preprocessing Data"
  modelData <- prepData dataset
  let tensorData = prepTensors modelData

  plotExampleData modelData tensorData

  let tIndices = asTensor (fipsIdxs modelData)
  let embedDim = 2
  weights <- randnIO' [M.size $ fipsMap modelData, 2]
  let locEmbed = embedding' weights tIndices
  print $ indexSelect' 0 [0 .. 10] locEmbed

  -- define fipsSpace
  let fipsList = M.keys . fipsMap $ modelData
  putStrLn "Number of counties:"
  print $ length fipsList

  let smallData = filterOn tFips (eq 1223) tensorData
  let cases = newCases (tCases smallData)
  let tsData = expandToSplits 1 cases
  model <- sample Simple1dSpec {lstm1dSpec = LSTMSpec {inputSize = 1, hiddenSize = 6}, mlp1dSpec = LinearSpec 6 1}
  let input = ones' [1, 1]

  let ((hidden, cell), output) = forward model ((pastObs tsData) !! 100)
  let actual = (nextObs tsData !! 100) !! 0
  print "output"
  print output

  print "loss"
  print $ mseLoss actual output

  putStrLn "Done"
