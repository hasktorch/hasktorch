{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Data.CsvDatastream
  ( BufferSize,
    NamedColumns (..),
    CsvDatastream' (..),
    CsvDatastream,
    CsvDatastreamNamed,
    csvDatastream,
    tsvDatastream,

    -- * Reexports
    FromField (..),
    FromRecord (..),
    FromNamedRecord (..),
  )
where

import qualified Control.Foldl as L
import Control.Monad
import Control.Monad.ST
import Data.Array.ST
import Data.Char (ord)
import Data.Csv (DecodeOptions (decDelimiter))
import Data.STRef
import Data.Vector (Vector)
import qualified Data.Vector as V
import Lens.Family (view)
import Pipes
import qualified Pipes.ByteString as B
import Pipes.Csv
import Pipes.Group (chunksOf, folds)
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import qualified Pipes.Safe.Prelude as Safe
import System.IO (IOMode (ReadMode))
import System.Random
import Torch.Data.StreamedPipeline

data NamedColumns = Unnamed | Named

type BufferSize = Int

-- TODO: implement more options

-- | A CSV datastream. The datastream instance of this type streams
-- samples of `batches` from a CSV file at the specified file path. Batches
-- are yielded in constant memory, but if shuffling is enabled, then there
-- will be at most @'BufferSize'@ records stored in memory.
data CsvDatastream' batches (named :: NamedColumns) = CsvDatastream'
  { -- | CSV file path.
    filePath :: FilePath,
    -- | Column delimiter.
    delimiter :: !B.Word8,
    -- | Does the file have a header?
    hasHeader :: HasHeader,
    -- | Batch size.
    -- , filter     :: Maybe (batches -> Bool)
    batchSize :: Int,
    -- | Buffered shuffle with specified buffer size.
    bufferedShuffle :: Maybe BufferSize,
    -- | Drop the last batch if it is less than batch size.
    dropLast :: Bool
  }

-- | A specialized version of CsvDatastream'. Use this type if you want to decode
-- a CSV file with records defined by the order of the columns.
type CsvDatastream batches = CsvDatastream' batches Unnamed

-- | A specialized version of CsvDatastream'. Use this type if you want to decode
-- a CSV file with records that have @'FromNamedRecord'@ instance. This decodes each field
-- of the record by the corresponding column with the given header name.
type CsvDatastreamNamed batches = CsvDatastream' batches Named

-- | Produce a CsvDatastream' from the given file with default options, and tab separated columns.
tsvDatastream :: forall (isNamed :: NamedColumns) batches. FilePath -> CsvDatastream' batches isNamed
tsvDatastream filePath = (csvDatastream filePath) {delimiter = fromIntegral $ ord '\t'}

-- | Produce a CsvDatastream' from the given file with default options, and comma separated columns.
csvDatastream :: forall (isNamed :: NamedColumns) batches. FilePath -> CsvDatastream' batches isNamed
csvDatastream filePath =
  CsvDatastream'
    { filePath = filePath,
      delimiter = fromIntegral $ ord ',',
      hasHeader = NoHeader,
      batchSize = 1,
      -- , filter = Nothing
      bufferedShuffle = Nothing,
      dropLast = True
    }

instance
  ( MonadBaseControl IO m,
    Safe.MonadSafe m,
    FromRecord batch
  ) =>
  Datastream m () (CsvDatastream batch) (Vector batch)
  where
  streamSamples csv@CsvDatastream' {..} _ = readCsv csv (decodeWith (defaultDecodeOptions {decDelimiter = delimiter}) hasHeader)

instance
  ( MonadBaseControl IO m,
    Safe.MonadSafe m,
    FromNamedRecord batch
  ) =>
  Datastream m () (CsvDatastreamNamed batch) (Vector batch)
  where
  streamSamples csv@CsvDatastream' {..} _ = readCsv csv (decodeByNameWith (defaultDecodeOptions {decDelimiter = delimiter}))

readCsv CsvDatastream' {..} decode = Select $
  Safe.withFile filePath ReadMode $ \fh ->
    -- this quietly discards errors in decoding right now, probably would like to log this
    if dropLast
      then streamRecords fh >-> P.filter (\v -> V.length v == batchSize)
      else streamRecords fh
  where
    streamRecords fh = case bufferedShuffle of
      Nothing -> L.purely folds L.vector $ view (chunksOf batchSize) $ decode (produceLine fh) >-> P.concat
      Just bufferSize ->
        L.purely folds L.vector $
          view (chunksOf batchSize) $
            (L.purely folds L.list $ view (chunksOf bufferSize) $ decode (produceLine fh) >-> P.concat) >-> shuffleRecords
    -- what's a good default chunk size?
    produceLine fh = B.hGetSome 1000 fh
    -- probably want a cleaner way of reyielding these chunks
    shuffleRecords = do
      chunks <- await
      std <- Torch.Data.StreamedPipeline.liftBase newStdGen
      mapM_ yield $ fst $ shuffle' chunks std

--  https://wiki.haskell.org/Random_shuffle
shuffle' :: [a] -> StdGen -> ([a], StdGen)
shuffle' xs gen =
  runST
    ( do
        g <- newSTRef gen
        let randomRST lohi = do
              (a, s') <- liftM (randomR lohi) (readSTRef g)
              writeSTRef g s'
              return a
        ar <- newArray n xs
        xs' <- forM [1 .. n] $ \i -> do
          j <- randomRST (i, n)
          vi <- readArray ar i
          vj <- readArray ar j
          writeArray ar j vi
          return vj
        gen' <- readSTRef g
        return (xs', gen')
    )
  where
    n = Prelude.length xs
    newArray :: Int -> [a] -> ST s (STArray s Int a)
    newArray n xs = newListArray (1, n) xs
