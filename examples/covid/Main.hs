{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import Control.Arrow (Arrow (first))
import Control.Monad ((>=>), foldM, when)
import Control.Monad.Cont (ContT (ContT), runCont, runContT)
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import Data.Map as M
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

instance FromRecord UsCounties

-- instance FromNamedRecord UsCounties

data Observations = Observations
  { time :: Tensor, -- date to a time point (integral values)
    location :: Tensor, -- fips indices representation -> embedding
    casesCount :: Tensor,
    deathCount :: Tensor
  }
  deriving (Eq, Show)

dat2Tensor :: V.Vector UsCounties -> IO Observations
dat2Tensor dataset = do
  day :: Day <- parseTimeM True defaultTimeLocale "%F" "2020-02-23"
  -- let x = (parseTimeM True defaultTimeLocale "%F") <$>  (V.map date dataset) -- TODO implement
  -- let ordinalDate = V.map toOrdinalDate day
  pure
    Observations
      { time = undefined,
        location = locEmbed,
        casesCount = undefined,
        deathCount = undefined
      }
  where
    idxFips = let fipsList = (V.uniq $ fmap fips dataset) in V.toList $ V.zip fipsList (V.fromList [1 .. (length fipsList)])
    idxMap = M.fromList idxFips
    indices = fmap ((M.!) idxMap) (fmap fips dataset)
    tIndices = asTensor . V.toList $ indices
    locEmbed = embedding' (onesLike tIndices) tIndices

--

modelTrain = undefined

-- TODO - use this
counties = (csvDataset @UsCounties "data/us-counties.csv") {batchSize = 4, shuffle = Nothing}

loadDataset fileName = do
  csvData <- BL.readFile fileName
  case decode HasHeader csvData of
    Left err -> error err
    Right (v :: V.Vector UsCounties) -> pure v

main :: IO ()
main = do
  -- let counties = (csvDataset @UsCounties "data/us-counties.csv") { batchSize = 4 , shuffle = Nothing}
  -- (countiesList) <- makeListT counties (Select $ yield ())

  dataset <- loadDataset "data/us-counties.csv"
  pPrint $ V.take 3 $ dataset
  let idxFips = let fipsList = (V.uniq $ fmap fips dataset) in V.toList $ V.zip fipsList (V.fromList [1 .. (length fipsList)])
  let idxMap = M.fromList idxFips
  let indices = fmap ((M.!) idxMap) (fmap fips dataset)
  let tIndices = asTensor . V.toList $ indices
  let locEmbed = embedding' (onesLike tIndices) tIndices

  -- pPrint $ dataset V.! 1
  putStrLn "Done"
