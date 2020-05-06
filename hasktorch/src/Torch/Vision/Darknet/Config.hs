{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Vision.Darknet.Config where

import Control.Monad (foldM, forM)
import Data.Either (isLeft, isRight, lefts, rights)
import Data.Ini.Config
import Data.Map (Map, lookup)
import Data.Sequence (Seq)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import GHC.Exts
import Prelude hiding (lookup)

type Index = Int

type InputChannels = Int

type OutputChannels = Int

data DarknetConfig' = DarknetConfig' GlobalConfig (Map Index LayerConfig) deriving (Show, Eq)

data DarknetConfig = DarknetConfig GlobalConfig (Map Index (LayerConfig, InputChannels, OutputChannels)) deriving (Show, Eq)

addChannels :: DarknetConfig' -> Either String DarknetConfig
addChannels cfg@(DarknetConfig' global layer_configs) = do
  v <- forM (toList layer_configs) $ \(idx, layer) -> do
    oc <- outputChannels cfg idx
    ic <- inputChannels cfg idx
    pure $ (idx, (layer, ic, oc))
  return (DarknetConfig global (fromList v))

outputChannels :: DarknetConfig' -> Int -> Either String Int
outputChannels cfg@(DarknetConfig' global layer_configs) idx =
  case lookup idx layer_configs of
    Nothing -> Left $ "Not found index:" ++ show idx ++ " in darnet"
    Just x ->
      case x of
        Convolution {..} -> pure filters
        Route {..} -> foldM (\a b -> (+ b) <$> outputChannels cfg a) 0 layers
        ShortCut {..} -> outputChannels cfg from
        _ -> inputChannels cfg idx

inputChannels :: DarknetConfig' -> Int -> Either String Int
inputChannels (DarknetConfig' global _) 0 = pure $ channels global
inputChannels cfg@(DarknetConfig' global layer_configs) idx =
  case lookup idx layer_configs of
    Nothing -> Left $ "Not found index:" ++ show idx ++ " in darnet"
    Just x ->
      case x of
        Route {..} -> outputChannels cfg idx
        ShortCut {..} -> outputChannels cfg idx
        _ -> outputChannels cfg (idx -1)

data GlobalConfig
  = Global
      { channels :: Int,
        height :: Int
      }
  deriving (Show, Eq)

data LayerConfig
  = Convolution
      { batch_normalize :: Bool,
        filters :: Int,
        layer_size :: Int,
        stride :: Int,
        activation :: String
      }
  | MaxPool
      { layer_size :: Int,
        stride :: Int
      }
  | UpSample
      { stride :: Int
      }
  | Route
      { layers :: [Int]
      }
  | ShortCut
      { from :: Int
      }
  | Yolo
      { mask :: [Int],
        anchors :: [(Int, Int)],
        classes :: Int
      }
  deriving (Show, Eq)

parseIntList :: T.Text -> Either String [Int]
parseIntList line = listWithSeparator "," number line

parseInt2List :: T.Text -> Either String [(Int, Int)]
parseInt2List line = do
  intlist <- listWithSeparator "," number line
  let loop [] _ = []
      loop (x : xs) Nothing = loop xs (Just x)
      loop (y : xs) (Just x) = (x, y) : loop xs Nothing
  return $ loop intlist Nothing

configParser' :: IniParser [Either GlobalConfig LayerConfig]
configParser' = (toList <$>) $ sectionsOf pure $ \section -> do
  case section of
    "net" ->
      (Left <$>) $
        Global
          <$> fieldOf "channels" number
          <*> fieldOf "height" number
    "convolutional" ->
      (Right <$>) $
        Convolution
          <$> ((== Just (1 :: Int)) <$> (fieldMbOf "batch_normalize" number))
          <*> fieldOf "filters" number
          <*> fieldOf "size" number
          <*> fieldOf "stride" number
          <*> fieldOf "activation" string
    "maxpool" ->
      (Right <$>) $
        MaxPool
          <$> fieldOf "size" number
          <*> fieldOf "stride" number
    "upsample" ->
      (Right <$>) $
        UpSample
          <$> fieldOf "stride" number
    "route" ->
      (Right <$>) $
        Route
          <$> fieldOf "layers" parseIntList
    "shortcut" ->
      (Right <$>) $
        ShortCut
          <$> fieldOf "from" number
    "yolo" ->
      (Right <$>) $
        Yolo
          <$> fieldOf "mask" parseIntList
          <*> fieldOf "anchors" parseInt2List
          <*> fieldOf "classes" number
    other -> error $ "Unknown darknet layer-type: " ++ show other

readIniFile :: String -> IO (Either String DarknetConfig)
readIniFile filepath = do
  contents <- T.readFile filepath
  case parseIniFile contents configParser' of
    Left error -> return $ Left error
    Right v -> do
      let configs = rights $ filter isRight v
          globalconfigs = lefts $ filter isLeft v
      if length globalconfigs == 0
        then return $ Left "net section is not defined."
        else
          if length globalconfigs > 1
            then return $ Left "net section is duplicated."
            else return $ addChannels $ DarknetConfig' (head globalconfigs) (fromList $ zip [0 ..] configs)
