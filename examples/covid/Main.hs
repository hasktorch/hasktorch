{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import CovidData
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
  let embedDim = 2
  weights <- randnIO' [M.size $ fipsMap modelData, 2]
  let locEmbed = embedding' weights tIndices
  print $ indexSelect' 0 [0 .. 10] locEmbed

  -- define fipsSpace
  let fipsList = M.keys . fipsMap $ modelData
  putStrLn "Number of counties:"
  print $ length fipsList

  initializedModel <- sample Simple1dSpec { lstm1dSpec = LSTMSpec {inputSize = 1, hiddenSize = 2} }
  -- let spec = optimSpec initializedModel undefined

  let smallData = filterOn tFips (eq 1223) tensorData
  let cases = clampMin 0.0 . diff . tCases $ smallData
  let casesList = reshape [1, 1] <$> asTensor <$> (fromIntegral <$> (asValue cases :: [Int]) :: [Float])
  model <- sample Simple1dSpec { lstm1dSpec = LSTMSpec {inputSize = 1, hiddenSize = 6} }
  let input = ones' [1, 1]
  let check = lstmCellForward (lstm1d model)  (zeros' [1, 6], zeros' [1, 6]) input


  print "check"
  print check

  -- let result = forward model tensorList
  print "result"
  -- print result

  putStrLn "Done"
