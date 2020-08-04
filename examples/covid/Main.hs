{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Control.Arrow (Arrow (first))
import Control.Monad ((>=>), foldM, when)
import Control.Monad.Cont (ContT (ContT), runCont, runContT)

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

import Torch
import Torch as T
import Torch.Data.CsvDataset
import Torch.Data.Pipeline (FoldM (FoldM))
-- import Torch.Data.StreamedPipeline (MonadBaseControl, pmap, makeListT)
import Torch.Data.StreamedPipeline

import CovidData
import CovidUtil
import TimeSeriesModel

modelTrain = undefined

-- TODO - use this
-- counties = (csvDataset @UsCounties "data/us-counties.csv") {batchSize = 4, shuffle = Nothing}

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

  -- define fipsSpace
  let fipsList = M.keys . fipsMap $ modelData

  -- check are there different numbers of points for different fips? yes
  print $ length $ Prelude.filter (\x -> x == 2102) (fipsIdxs modelData)
  print $ length $ Prelude.filter (\x -> x == 739) (fipsIdxs modelData)

  -- pPrint $ filterOn tTimes (eq $ asTensor (20 :: Int)) tensorData 
  -- pPrint $ filterOn tFips (eq $ asTensor (1 :: Int)) tensorData

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
