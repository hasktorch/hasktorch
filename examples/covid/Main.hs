{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Data.Maybe (fromJust)
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

main :: IO ()
main = do
  putStrLn "Loading Data"
  dataset <- loadDataset "data/us-counties.csv"
  putStrLn "Preprocessing Data"
  modelData <- prepData dataset
  let tensorData = prepTensors modelData

  plotExampleData modelData tensorData

  let tIndices = asTensor (fipsIdxs modelData)
      embedDim = 2
  weights <- randnIO' [M.size $ fipsMap modelData, 2]
  let locEmbed = embedding' weights tIndices
  print $ indexSelect' 0 [0 .. 10] locEmbed

  -- define fipsSpace
  let fipsList = M.keys . fipsMap $ modelData
  putStrLn "Number of counties:"
  print $ length fipsList

  let smallData = filterOn tFips (eq 1223) tensorData
      cases = newCases (tCases smallData)
      tsData = expandToSplits 1 cases
  model <- sample Simple1dSpec {lstm1dSpec = LSTMSpec {inputSize = 1, hiddenSize = 6}, mlp1dSpec = LinearSpec 6 1}
  let input = ones' [1, 1]

  -- manual retrieval
  let ((hidden, cell), output) = forward model (getObs' 100 (pastObs tsData))
  print output
  print $ mseLoss (getTime' 100 0 (nextObs tsData)) output

  -- use data loading mechanism
  let (past, future) = getItem tsData 100 1
  let ((hidden, cell), output) = forward model (getObs' 0 past)
  print output
  print $ mseLoss (getTime' 0 0 future) output

  putStrLn "Done"
