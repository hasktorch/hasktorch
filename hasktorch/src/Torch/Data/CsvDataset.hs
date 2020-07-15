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

import System.Random
import Data.Array.ST
import Control.Monad
import Control.Monad.ST
import Data.STRef

import           Torch.Typed
import qualified Torch.DType as D
import           Data.Reflection (Reifies(reflect))
import           Data.Proxy (Proxy(Proxy))
import           GHC.Exts (IsList(fromList))
import           Control.Monad 
import qualified Data.Vector as V
import qualified Torch.Tensor as D
import           GHC.TypeLits (KnownNat)
import           Torch.Data.StreamedPipeline
import           Pipes.Safe

import qualified Control.Foldl as L
import           Control.Foldl.Text (Text)
import           Control.Monad.Base (MonadBase)
import           Data.ByteString (hGetLine, hGetContents)
import           Data.Set.Ordered as OSet hiding (fromList)
import           Lens.Family (view)
import           Pipes (liftIO, ListT(Select), yield, (>->), await)
import qualified Pipes.ByteString as B
import           Pipes.Csv
import           Pipes.Group (takes, folds, chunksOf)
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import qualified Pipes.Safe.Prelude as Safe
import           System.IO (IOMode(ReadMode))
import Pipes.Concurrent (unbounded)
import Control.Monad.Trans.Control (MonadBaseControl)


-- instance FromField
instance FromField a => FromField [a] where
  -- simply wrap a single 'a' into a list
  parseField = fmap pure . parseField 

instance ( KnownNat seqLen
         , KnownDevice device
         , FromField [Int]
         )
    => FromRecord (Tensor device 'Int64 '[1, seqLen]) where
  parseRecord 
    s | V.length s < natValI @seqLen = mzero
      | otherwise = fromList <$> (parseRecord $ V.take (natValI @seqLen) s ) >>= \case
            Nothing -> mzero 
            Just s -> pure s

  -- these two instances actually don't make sense right now
  -- since fields only work between each delimiter
instance ( KnownNat seqLen
         , KnownDevice device
         , FromField [Float]
          )
    => FromRecord (Tensor device 'D.Float '[1, seqLen]) where
  parseRecord 
    s | V.length s < natValI @seqLen = mzero
      | otherwise = fromList <$> (parseRecord $ V.take (natValI @seqLen) s ) >>= \case
            Nothing -> mzero 
            Just s -> pure s

data NamedColumns = Unnamed | Named
type BufferSize = Int

data CsvDataset batches = CsvDataset { filePath :: FilePath
                                     , decDelimiter :: !B.Word8
                                     , byName :: NamedColumns
                                     , hasHeader :: HasHeader
                                     , batchSize :: Int
                                     , filter :: Maybe (batches -> Bool)
                                     , numBatches :: Maybe Int
                                     , shuffle :: Maybe BufferSize
                                     }

csvDataset :: forall batches . FilePath -> CsvDataset batches
csvDataset filePath  = CsvDataset { filePath = filePath
                                  , decDelimiter = 44 -- comma
                                  , byName = Unnamed
                                  , hasHeader = NoHeader
                                  , batchSize = 1
                                  , filter = Nothing
                                  , numBatches = Nothing
                                  , shuffle = Nothing
                                  }

instance ( MonadPlus m
         , MonadBase IO m
         , MonadBaseControl IO m
         , Safe.MonadSafe m
         , FromRecord batch -- these constraints make CsvDatasets only able to parse records, might not be the best idea
         , FromNamedRecord batch
         -- , Monoid batch
         ) => Datastream m () (CsvDataset batch) [batch] where
  streamBatch CsvDataset{..} _ = Select $ Safe.withFile filePath ReadMode $ \fh ->
    -- this quietly discards errors right now, probably would like to log this
    -- TODO: optionally drop last chunk if it's less than batchSize (would want to use vectors for O(1) length) 
    -- TODO: optionally take a fixed number of batches
    -- TODO: we could concurrently stream in records, and batch records in another thread
    case shuffle of
      Nothing -> L.purely folds L.list $ view (chunksOf batchSize) $ decodeRecords fh >-> P.concat
      Just bufferSize -> L.purely folds L.list $ view (chunksOf batchSize) $
         (L.purely folds L.list $ view (chunksOf bufferSize) $ decodeRecords fh >-> P.concat) >-> shuffleRecords
    where
      decodeRecords fh = case byName of
                           Unnamed -> decode hasHeader (produceLine fh)
                           Named   -> decodeByName (produceLine fh)
      -- what's a good default chunk size? 
      produceLine fh = B.hGetSome 1000 fh
      -- probably want a cleaner way of reyielding these chunks
      shuffleRecords = do
        chunks <- await 
        std <- Torch.Data.StreamedPipeline.liftBase getStdGen
        mapM_ yield  $ (fst $  shuffle' chunks std)
        
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
