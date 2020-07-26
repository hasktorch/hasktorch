{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.ByteString.Lazy as BL
import Control.Arrow (Arrow(first))
import GHC.Generics (Generic)
import Data.Csv
import qualified Data.Vector as V
import Pipes
import Torch
import Torch.Data.CsvDataset
import Torch.Data.Pipeline (FoldM (FoldM))
import Torch.Data.StreamedPipeline (MonadBaseControl, pmap, makeListT)

data UsCounties = UsCounties {
    date :: String,
    county :: String,
    state :: String,
    fips :: String,
    cases :: Int,
    deaths :: Maybe Int,
    confirmed_cases :: Maybe Int,
    confirmed_deaths :: Maybe Int,
    probable_cases :: Maybe Int,
    probable_deaths :: Maybe Int
} deriving (Eq, Generic, Show)

instance FromRecord UsCounties

dat2Tensor = undefined

modelTrain = undefined

main = do
  -- let counties = (csvDataset @UsCounties "data/us-counties.csv") { batchSize = 4 , shuffle = Nothing}
  csvData <- BL.readFile "data/us-counties.csv"
  case decode HasHeader csvData of
        Left err -> putStrLn err
        Right (v :: V.Vector UsCounties) -> V.forM_ v $ \ record ->
          putStrLn (county record)


  putStrLn "Done"