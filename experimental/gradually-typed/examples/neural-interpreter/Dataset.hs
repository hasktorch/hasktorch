{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TemplateHaskell #-}

module Dataset where

import Control.Concurrent.STM (atomically)
import Control.Concurrent.STM.TVar (TVar, modifyTVar', readTVar)
import Control.Monad (guard)
import Control.Monad.State (MonadIO (liftIO), evalStateT, runState)
import Data.Aeson.TH (defaultOptions, deriveJSON)
import Data.HashSet (HashSet)
import qualified Data.HashSet as HashSet
import Data.Hashable (Hashable)
import qualified Data.List as List
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Text (Text)
import GHC.Generics (Generic)
import qualified Gen
import qualified Hedgehog.Internal.Gen as Gen
import Hedgehog.Internal.Seed (Seed)
import qualified Hedgehog.Internal.Seed as Seed
import qualified Pipes.Safe as P
import qualified STLC
import Torch.GraduallyTyped
import Control.Monad.Reader (ReaderT, MonadReader (ask))

type Tokenizer = String -> IO [Int]

type Detokenizer = [Int] -> IO String

data STLCData = STLCData
  { name :: Text,
    seeds :: Set Seed,
    targetNfSteps :: Maybe (Set Int),
    maxInputLength :: Int,
    maxTargetLength :: Int,
    tokenize :: Tokenizer,
    detokenize :: Detokenizer
  }

data STLCExample a = STLCExample
  { exTy :: !STLC.Ty,
    exInputExp :: !(STLC.Exp a),
    exInputPPrint :: !String,
    exInputIds :: ![Int],
    exDecodedInputIds :: !String,
    exTargetExp :: !(STLC.Exp a),
    exTargetNfSteps :: !Int,
    exTargetPPrint :: !String,
    exTargetIds :: ![Int],
    exDecodedTargetIds :: !String
  }
  deriving stock (Show, Eq, Ord, Generic)
  deriving anyclass (Hashable)

$(deriveJSON defaultOptions ''STLCExample)

mkExample ::
  Tokenizer ->
  Detokenizer ->
  Maybe (Set Int) ->
  Int ->
  Int ->
  Seed.Seed ->
  ReaderT (TVar (HashSet [Int])) (P.SafeT IO) (STLCExample Int)
mkExample tokenize detokenize targetNfSteps maxInputLength maxTargetLength seed = flip evalStateT seed . Gen.sample' $ do
  exTy <- Gen.genTy
  exInputExp <- Gen.generalize $ Gen.genWellTypedExp exTy

  let (exTargetExp, exTargetNfSteps) = flip runState 0 $ STLC.nf exInputExp
  guard (maybe True (\s -> exTargetNfSteps `Set.member` s) targetNfSteps)

  let exInputPPrint = STLC.pprint exInputExp
  exInputIds <- liftIO . tokenize $ exInputPPrint <> "</s>"
  guard (List.length exInputIds <= maxInputLength)

  cacheHit <- do
    tvar <- ask
    liftIO . atomically $ do
      dedupCache <- readTVar tvar
      pure $ HashSet.member exInputIds dedupCache
  guard (not cacheHit)

  () <- do
    tvar <- ask
    liftIO . atomically $ modifyTVar' tvar (HashSet.insert exInputIds)

  let exTargetPPrint = STLC.pprint exTargetExp
  exTargetIds <- liftIO . tokenize $ exTargetPPrint <> "</s>"
  guard (List.length exTargetIds <= maxTargetLength)

  exDecodedInputIds <- liftIO $ detokenize exInputIds
  exDecodedTargetIds <- liftIO $ detokenize exTargetIds

  pure STLCExample {..}

instance Dataset (ReaderT (TVar (HashSet [Int])) (P.SafeT IO)) STLCData Seed (STLCExample Int) where
  getItem STLCData {..} seed = do
    guard $ Set.member seed seeds
    mkExample tokenize detokenize targetNfSteps maxInputLength maxTargetLength seed
  keys STLCData {..} = seeds
