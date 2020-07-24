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
module Torch.Data.CsvDataset ( CsvDataset(..)
                             , csvDataset
                             , NamedColumns
                             , FromField(..)
                             , FromRecord(..)
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
import           Pipes (liftIO, ListT(Select), yield, (>->), await)
import qualified Pipes.ByteString as B
import           Pipes.Csv
import           Pipes.Group (takes, folds, chunksOf)
import           Pipes.Safe
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import qualified Pipes.Safe.Prelude as Safe
import           System.IO (IOMode(ReadMode))

data NamedColumns = Unnamed | Named
type BufferSize = Int

-- TODO: maybe we use a type family to specify if we want to decode using named or unnamed!
-- TODO: implement more options
data CsvDataset batches = CsvDataset { filePath   :: FilePath
                                     , delimiter  :: !B.Word8
                                     , byName     :: NamedColumns
                                     , hasHeader  :: HasHeader
                                     , batchSize  :: Int
                                     , filter     :: Maybe (batches -> Bool)
                                     , shuffle    :: Maybe BufferSize
                                     , dropLast   :: Bool
                                     }
tsvDataset :: forall batches . FilePath -> CsvDataset batches
tsvDataset filePath = (csvDataset filePath) { delimiter = fromIntegral $ ord '\t' }

csvDataset :: forall batches . FilePath -> CsvDataset batches
csvDataset filePath  = CsvDataset { filePath = filePath
                                  , delimiter = fromIntegral $ ord ','
                                  , byName = Unnamed
                                  , hasHeader = NoHeader
                                  , batchSize = 1
                                  , filter = Nothing
                                  , shuffle = Nothing
                                  , dropLast = True
                                  }

instance ( MonadPlus m
         , MonadBase IO m
         , MonadBaseControl IO m
         , Safe.MonadSafe m
         -- these constraints make CsvDatasets only able to parse records, might not be the best idea
         , FromRecord batch 
         , FromNamedRecord batch
         ) => Datastream m () (CsvDataset batch) (Vector batch) where
  streamBatch CsvDataset{..} _ = Select $ Safe.withFile filePath ReadMode $ \fh ->
    -- this quietly discards errors right now, probably would like to log this
    -- TODO:  we could concurrently stream in records, and batch records in another thread
    -- actually this ^ isn't possible since this is in the Proxy monad, instead we should
    -- be able to define a transform which collates in parallel, and yield batch size 1 here
    if dropLast
    then streamRecords fh >-> P.filter (\v -> V.length v == batchSize)
    else streamRecords fh

    where
      streamRecords fh = case shuffle of
        Nothing -> L.purely folds L.vector $ view (chunksOf batchSize) $ decodeRecords fh >-> P.concat
        Just bufferSize -> L.purely folds L.vector
                         $ view (chunksOf batchSize)
                         $ ( L.purely folds L.list
                            $ view (chunksOf bufferSize)
                            $ decodeRecords fh >-> P.concat
                           )
                         >-> shuffleRecords

      decodeRecords fh = case byName of
                           Unnamed -> decodeWith (defaultDecodeOptions { decDelimiter = delimiter }) hasHeader (produceLine fh)
                           Named   -> decodeByName (produceLine fh)
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

