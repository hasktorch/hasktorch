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
module Torch.Data.CsvDataset where

import           Torch.Typed
import qualified Torch.DType as D
import           Data.Reflection (Reifies(reflect))
import           Data.Proxy (Proxy(Proxy))
import           GHC.Exts (IsList(fromList))
import           Control.Monad (MonadPlus(mzero))
import qualified Data.Vector as V
import qualified Torch.Tensor as D
import           GHC.TypeLits (KnownNat)
import           Torch.Data.StreamedPipeline
import           Pipes.Safe

import qualified Control.Foldl as L
import           Control.Foldl.Text (Text)
import           Control.Monad.Base (MonadBase)
import           Data.ByteString (hGetLine)
import           Data.Set.Ordered as OSet hiding (fromList)
import           Lens.Family (view)
import           Pipes (liftIO, ListT(Select), yield, (>->))
import qualified Pipes.ByteString as B
import           Pipes.Csv
import           Pipes.Group (takes, folds, chunksOf)
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import qualified Pipes.Safe.Prelude as Safe
import           System.IO (IOMode(ReadMode))


  -- these two instances actually don't make sense right now
  -- since fields only work between each delimiter
instance ( KnownNat seqLen
         , KnownDevice device
         , FromField [Int]
         )
    => FromRecord (Tensor device 'Int64 '[1, seqLen]) where
  parseRecord 
    s | V.length s < natValI @seqLen = mzero
      | otherwise = fromList <$> (parseRecord  $ V.take (natValI @seqLen) s ) >>= \case
            Nothing -> mzero 
            Just s -> pure s

instance ( KnownNat seqLen
         , KnownDevice device
         , FromField [Float]
          )
    => FromRecord (Tensor device 'D.Float '[1, seqLen]) where
  parseRecord 
    s | V.length s < natValI @seqLen = mzero
      | otherwise = fromList <$> (parseRecord  $ V.take (natValI @seqLen) s ) >>= \case
            Nothing -> mzero 
            Just s -> pure s

data NamedColumns = Unnamed | Named
data CsvDataset batches = CsvDataset { filePath :: FilePath
                                     , decDelimiter :: !B.Word8
                                     , byName :: NamedColumns
                                     , hasHeader :: HasHeader
                                     , batchSize :: Int
                                     }

instance ( MonadPlus m
         , MonadBase IO m
         , Safe.MonadSafe m
         , FromRecord batch -- these constraints make CsvDatasets only able to parse records, might not be the best idea
         , FromNamedRecord batch
         , Monoid batch
           
         ) => Datastream m (CsvDataset batch) batch where
  streamBatch CsvDataset{..} seed = Select $
    Safe.withFile filePath ReadMode $
    \fh -> do
      L.purely folds L.mconcat $ view (chunksOf batchSize) $
        (
          (\p -> case byName of
                  Unnamed -> decode hasHeader p
                  Named   -> decodeByName p
         )
        $ produceLine fh
        )
        >-> P.concat -- this quietly discards errors right now, probably would like to log this

        where produceLine fh =  (liftIO $ hGetLine fh) >>= yield  -- lines aren't streamed in right now, they probably should be

