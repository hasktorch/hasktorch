{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where
import           Torch.Data.StreamedPipeline
import           Torch.Data.CsvDataset
import           GHC.Generics
import           Data.Csv
import qualified Pipes.Prelude as P
import           Pipes.Safe (runSafeT)
import           Pipes
import Torch.Tensor (AsTensors)
import Torch (AsTensors(toTensors))


  -- We've written a FromField instance for [a] that behaves as the FromField instance
  -- of 'a'. This lets us use the monoidal instance for [].
data CsvRecord = CsvRecord { field1 :: [Int]
                           , field2 :: [Int]
                           , field3 :: [Int]
                           } deriving (Eq, Show, Generic)
-- instance AsTensors CsvRecord where
  
instance FromRecord CsvRecord where
instance FromNamedRecord CsvRecord where
instance Semigroup CsvRecord where
  (<>) record1 record2 =  CsvRecord { field1 = field1 record1 <> field1 record2,
                                      field2 = field2 record1 <> field2 record2,
                                      field3 = field3 record1 <> field3 record2
                                      } 
instance Monoid CsvRecord where
  mempty = CsvRecord { field1 = mempty, field2 = mempty, field3 = mempty}

main :: IO ()
main = runSafeT $ do
 let dataset :: CsvDataset CsvRecord
     dataset = (csvDataset @CsvRecord "examples.csv") { batchSize = 10 }
 listT <- makeListT @_ @_ @_ @CsvRecord defaultDataloaderOpts dataset id (Select $ yield ())
 runEffect $ enumerate listT >->  P.chain (const $ liftIO $ print "here") >-> P.print

  
