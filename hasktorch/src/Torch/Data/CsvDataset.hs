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
-- import           Data.Set.Ordered as OSet hiding (fromList)
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
import Data.Vector (Vector)
import Data.Csv (DecodeOptions(decDelimiter))
import Data.Char (ord)



  
-- instance FromField a => FromField [a] where
--   -- simply wrap a single 'a' into a list
--   parseField = fmap pure . parseField 

-- instance ( KnownNat seqLen
--          , KnownDevice device
--          , FromField [Int]
--          )
--     => FromRecord (Tensor device 'Int64 '[1, seqLen]) where
--   parseRecord 
--     s | V.length s < natValI @seqLen = mzero
--       | otherwise = fromList <$> (parseRecord $ V.take (natValI @seqLen) s ) >>= \case
--             Nothing -> mzero 
--             Just s -> pure s

--   -- these two instances actually don't make sense right now
--   -- since fields only work between each delimiter
-- instance ( KnownNat seqLen
--          , KnownDevice device
--          , FromField [Float]
--           )
--     => FromRecord (Tensor device 'D.Float '[1, seqLen]) where
--   parseRecord 
--     s | V.length s < natValI @seqLen = mzero
--       | otherwise = fromList <$> (parseRecord $ V.take (natValI @seqLen) s ) >>= \case
--             Nothing -> mzero 
--             Just s -> pure s

data NamedColumns = Unnamed | Named
type BufferSize = Int


  -- TODO: maybe we use a type family to specify if we want to decode using named or unnamed!
data CsvDataset batches = CsvDataset { filePath :: FilePath
                                     , delimiter :: !B.Word8
                                     , byName       :: NamedColumns
                                     , hasHeader    :: HasHeader
                                     , batchSize  :: Int
                                     , filter     :: Maybe (batches -> Bool)
                                     -- , numBatches :: Maybe Int
                                     , shuffle    :: Maybe BufferSize
                                     , dropLast   :: Bool
                                     }
tsvDataset :: forall batches . FilePath -> CsvDataset batches
tsvDataset filePath = (csvDataset filePath) { delimiter = fromIntegral $ ord '\t' }

csvDataset :: forall batches . FilePath -> CsvDataset batches
csvDataset filePath  = CsvDataset { filePath = filePath
                                  , delimiter = 44 -- comma
                                  , byName = Unnamed
                                  , hasHeader = NoHeader
                                  , batchSize = 1
                                  , filter = Nothing
                                  -- , numBatches = Nothing
                                  , shuffle = Nothing
                                  , dropLast = True
                                  }

instance ( MonadPlus m
         , MonadBase IO m
         , MonadBaseControl IO m
         , Safe.MonadSafe m
         , FromRecord batch -- these constraints make CsvDatasets only able to parse records, might not be the best idea
         -- , FromNamedRecord batch
         -- , Monoid batch
         -- ) => Datastream m () (CsvDataset batch) [batch] where
         ) => Datastream m () (CsvDataset batch) (Vector batch) where
  streamBatch CsvDataset{..} _ = Select $ Safe.withFile filePath ReadMode $ \fh ->
    -- this quietly discards errors right now, probably would like to log this
    -- TODO:  we could concurrently stream in records, and batch records in another thread 

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
                           -- Named   -> decodeByName (produceLine fh)
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

-- \" \ 0.0010318 0.31201 -0.59768 -0.12583 -0.27524 0.29145 -0.30431 0.037122 0.94468 0.088085 -0.096273 0.40542 -0.6524 0.37716 0.53001 -0.30819 -0.27478 0.81041 0.53635 0.08759 0.35288 2.8857 -0.12505 -0.035968 0.1355 -0.29932 0.56154 -0.59429 -0.34991 0.68559 -0.11537 0.088894 -0.29829 -0.20768 0.6911 0.068548 -0.50814 -0.97722 0.035782 -0.54053 -0.16178 0.47542 -0.71652 0.31939 -0.049291 0.32565 0.5551 -0.42967 1.0137 0.30304 0.44503 0.46987 0.074292 -0.35762 -0.13736 -0.16066 -0.28606 0.00012358 -0.043422 -0.36702 0.40895 -0.40467 -0.51635 0.28044 -0.27625 -0.20352 0.084837 0.078145 0.32895 0.20482 0.17551 -0.2309 -0.066328 0.46467 -0.38932 -0.23018 -0.098122 -0.12376 -0.028503 0.040342 0.71158 -0.27804 0.10137 0.51113 -0.42648 -0.12535 -0.39139 0.29972 0.1868 -0.58113 0.38192 0.1654 0.33566 -0.47696 -0.24135 -0.33114 0.34815 -0.087061 -0.16406 -0.1023 0.53301 0.45604 0.33281 -0.33851 0.30248 -0.15048 -0.33033 0.6668 -0.28609 1.0374 -0.70909 -0.084268 0.47584 -0.16125 0.19906 0.51991 0.026676 -0.25567 -0.32363 0.13236 0.54006 0.093419 0.12268 -0.62967 0.00074921 -0.53441 0.25609 0.39963 0.80527 -0.41176 0.18955 0.024526 -0.20222 0.33781 0.007872 -0.64424 -0.5309 0.027875 0.088964 0.29687 -0.038186 0.77967 -0.069213 -0.58387 1.1816 0.60791 -0.4021 -0.32223 -0.071283 -0.022764 -0.46393 0.99797 -0.15308 0.4712 0.73921 -0.12487 -0.38275 -0.39598 0.12852 -0.4951 0.19752 -0.25436 0.16616 -0.18135 0.16374 -0.16194 -0.32013 -0.33111 0.53662 0.53301 -0.32166 -0.64362 1.0576 -0.18019 -0.40932 -0.16084 -1.0094 0.22972 0.5431 0.05891 1.7502 -0.19753 -0.05401 0.016083 -0.55523 -0.20231 -0.32613 -0.38783 0.61428 -0.11216 0.42319 -0.44729 -0.35638 -0.32698 -0.12662 -0.28858 0.08092 0.14493 0.052563 0.75007
