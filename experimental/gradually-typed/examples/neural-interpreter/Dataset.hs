{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Dataset where

import Control.Monad (guard)
import Control.Monad.State (MonadIO (liftIO), evalStateT)
import qualified Data.List as List
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import GHC.Generics (Generic)
import qualified Gen
import qualified Hedgehog.Internal.Gen as Gen
import Hedgehog.Internal.Seed (Seed)
import qualified Hedgehog.Internal.Seed as Seed
import qualified STLC
import qualified Tokenizers
import Torch.GraduallyTyped

data STLCData = STLCData
  { name :: Text,
    seeds :: Set Seed,
    maxInputLength :: Int,
    maxTargetLength :: Int,
    tokenizer :: Tokenizers.Tokenizer
  }

data STLCExample a = STLCExample
  { exTy :: !STLC.Ty,
    exInputExp :: !(STLC.Exp a),
    exInputPPrint :: !String,
    exInputIds :: ![Int],
    exDecodedInputIds :: !String,
    exTargetExp :: !(STLC.Exp a),
    exTargetPPrint :: !String,
    exTargetIds :: ![Int],
    exDecodedTargetIds :: !String
  }
  deriving stock (Show, Eq, Ord, Generic)

mkExample :: Tokenizers.Tokenizer -> Int -> Int -> Seed.Seed -> IO (STLCExample Int)
mkExample tokenizer maxInputLength maxTargetLength seed = flip evalStateT seed . Gen.sample' $ do
  exTy <- Gen.genTy
  exInputExp <- Gen.generalize $ Gen.genWellTypedExp exTy
  let exInputPPrint = STLC.pprint exInputExp
      exTargetExp = STLC.nf exInputExp
      exTargetPPrint = STLC.pprint exTargetExp
  exInputEnc <- liftIO $ Tokenizers.encode tokenizer (exInputPPrint <> "</s>")
  exInputIds <- liftIO $ Tokenizers.getIDs exInputEnc
  guard (List.length exInputIds <= maxInputLength)
  exTargetEnc <- liftIO $ Tokenizers.encode tokenizer (exInputPPrint <> "</s>")
  exTargetIds <- liftIO $ Tokenizers.getIDs exTargetEnc
  guard (List.length exTargetIds <= maxTargetLength)
  exDecodedInputIds <- liftIO $ Tokenizers.decode tokenizer exInputIds
  exDecodedTargetIds <- liftIO $ Tokenizers.decode tokenizer exTargetIds
  -- liftIO $ do
  --   putStrLn $ exInputPPrint <> " -> " <> exTargetPPrint
  --   putStrLn $ exDecodedInputIds <> " -> " <> exDecodedTargetIds
  pure STLCExample {..}

instance Dataset IO STLCData Seed (STLCExample Int) where
  getItem STLCData {..} seed = do
    guard $ Set.member seed seeds
    mkExample tokenizer maxInputLength maxTargetLength seed
  keys STLCData {..} = seeds

withTokenizer :: (Tokenizers.Tokenizer -> IO a) -> IO a
withTokenizer =
  Tokenizers.withTokenizerFromConfigFile
    "/tmp/t5-small-tokenizer.json"

testExample :: IO (STLCExample Int)
testExample = do
  withTokenizer $ \tokenizer -> do
    let maxInputLength = 512
        maxTargetLength = 512
        seed = Seed.from 45
    mkExample tokenizer maxInputLength maxTargetLength seed
