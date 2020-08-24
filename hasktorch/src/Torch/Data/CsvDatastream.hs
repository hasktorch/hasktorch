{-# LANGUAGE GADTs #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
module Torch.Data.CsvDatastream ( csvDatastream
                                , tsvDatastream 
                                , CsvDatastream
                                , CsvDatastreamNamed
                                , CsvDatastream'(..)
                                , FromField(..)
                                , FromRecord(..)
                                , FromNamedRecord(..)
                                ) where


import           System.Random
import           Data.Array.ST
import           Control.Monad
import           Control.Monad.ST
import           Data.STRef

import qualified Data.Vector as V
import           GHC.TypeLits (KnownNat)
import           Torch.Data.StreamedPipeline

import qualified Control.Foldl as L
import           Control.Monad.Base (MonadBase)
import           Control.Monad.Trans.Control (MonadBaseControl)
import           Data.ByteString (hGetLine, hGetContents)
import           Data.Char (ord)
import           Data.Csv (DecodeOptions(decDelimiter))
import           Data.Vector (Vector)
import           Lens.Family (view)
import           Pipes
import qualified Pipes.ByteString as B
import           Pipes.Csv
import           Pipes.Group (takes, folds, chunksOf)
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import qualified Pipes.Safe.Prelude as Safe
import           System.IO (IOMode(ReadMode))

data NamedColumns = Unnamed | Named
type BufferSize = Int

-- TODO: implement more options
data CsvDatastream' batches (named :: NamedColumns) = CsvDatastream' { filePath   :: FilePath
                                                                  , delimiter  :: !B.Word8
                                                                  , hasHeader  :: HasHeader
                                                                  , batchSize  :: Int
                                                                  -- , filter     :: Maybe (batches -> Bool)
                                                                  , bufferedShuffle    :: Maybe BufferSize
                                                                  , dropLast   :: Bool
                                                                  }

-- | Use if you want to decode with a record that has a FromRecord instance.
type CsvDatastream batches = CsvDatastream' batches Unnamed

-- | Use CsvDatasetNamed if you want to decode with a record that has FromNamedRecord instance,
-- | decoding fields of a given name from a csv file,
type CsvDatastreamNamed batches = CsvDatastream' batches Named

tsvDatastream :: forall (isNamed :: NamedColumns) batches . FilePath -> CsvDatastream' batches isNamed 
tsvDatastream filePath = (csvDatastream filePath) { delimiter = fromIntegral $ ord '\t' }

csvDatastream :: forall (isNamed :: NamedColumns) batches . FilePath -> CsvDatastream' batches isNamed
csvDatastream filePath = CsvDatastream' { filePath = filePath
                                     , delimiter = fromIntegral $ ord ','
                                     , hasHeader = NoHeader
                                     , batchSize = 1
                                     -- , filter = Nothing
                                     , bufferedShuffle = Nothing
                                     , dropLast = True
                                     }
                       

instance ( MonadBaseControl IO m
         , Safe.MonadSafe m
         , FromRecord batch 
         ) => Datastream m () (CsvDatastream batch) (Vector batch) where
  streamBatch csv@CsvDatastream'{..} _ = readCsv csv (decodeWith (defaultDecodeOptions { decDelimiter = delimiter }) hasHeader)

instance ( MonadBaseControl IO m
         , Safe.MonadSafe m
         , FromNamedRecord batch
         ) => Datastream m () (CsvDatastreamNamed batch) (Vector batch) where
  streamBatch csv@CsvDatastream'{..} _ = readCsv csv (decodeByNameWith (defaultDecodeOptions { decDelimiter = delimiter }))

readCsv CsvDatastream'{..} decode = Select $ Safe.withFile filePath ReadMode $ \fh ->
    -- this quietly discards errors in decoding right now, probably would like to log this
    if dropLast
    then streamRecords fh >-> P.filter (\v -> V.length v == batchSize)
    else streamRecords fh

    where
      streamRecords fh = case bufferedShuffle of
        Nothing -> L.purely folds L.vector $ view (chunksOf batchSize) $ decode (produceLine fh) >-> P.concat
        Just bufferSize -> L.purely folds L.vector
                         $ view (chunksOf batchSize)
                         $ (L.purely folds L.list $ view (chunksOf bufferSize) $ decode (produceLine fh) >-> P.concat) >-> shuffleRecords
      -- what's a good default chunk size? 
      produceLine fh = B.hGetSome 1000 fh
      -- probably want a cleaner way of reyielding these chunks
      shuffleRecords = do
        chunks <- await 
        std <- Torch.Data.StreamedPipeline.liftBase newStdGen
        mapM_ yield  $ fst $  shuffle' chunks std
        
--  https://wiki.haskell.org/Random_shuffle
shuffle' :: [a] -> StdGen -> ([a],StdGen)
shuffle' xs gen = runST (do
        g <- newSTRef gen
        let randomRST lohi = do
              (a,s') <- liftM (randomR lohi) (readSTRef g)
              writeSTRef g s'
              return a
        ar <- newArray n xs
        xs' <- forM [1..n] $ \i -> do
                j <- randomRST (i,n)
                vi <- readArray ar i
                vj <- readArray ar j
                writeArray ar j vi
                return vj
        gen' <- readSTRef g
        return (xs',gen'))
  where
    n = Prelude.length xs
    newArray :: Int -> [a] -> ST s (STArray s Int a)
    newArray n xs =  newListArray (1,n) xs
