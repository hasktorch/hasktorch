{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Vision.Darknet where

import Control.Monad (mapM)
import Data.Either
import Data.Ini.Config
import Data.List ((!!))
import Data.Maybe (fromMaybe)
import Data.Sequence (Seq)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import GHC.Exts
import GHC.Generics
import Torch.Autograd
import Torch.Functional
import qualified Torch.Functional.Internal as I
import Torch.Initializers
import Torch.NN
import Torch.Tensor hiding (size)
import Torch.TensorFactories

data GlobalSpec
  = GlobalSpec
      { channels :: Int,
        height :: Int
      }
  deriving (Show, Eq)

data LayerSpec
  = ConvolutionSpec
      { batch_normalize :: Bool,
        filters :: Int,
        size :: Int,
        stride :: Int,
        activation :: String
      }
  | MaxPoolSpec
      { size :: Int,
        stride :: Int
      }
  | UpSampleSpec
      { stride :: Int
      }
  | RouteSpec
      { layers :: [Int]
      }
  | ShortCutSpec
      { from :: Int
      }
  | YoloSpec
      { mask :: [Int],
        anchors :: [Int],
        classes :: Int
      }
  deriving (Show, Eq)

data DarknetSpec = DarknetSpec GlobalSpec [LayerSpec]
  deriving (Show)

data Layer
  = Convolution
      { weight :: Parameter,
        bias :: Parameter,
        func :: Tensor -> Tensor
      }
  | MaxPool
      { func :: Tensor -> Tensor
      }
  | UpSample
      { func :: Tensor -> Tensor
      }
  | Route
      { route :: [Tensor] -> Tensor
      }
  | ShortCut
      { shortcut :: [Tensor] -> Tensor
      }
  | Yolo
      { func :: Tensor -> Tensor
      }
  deriving (Generic)

instance Parameterized Layer

instance Parameterized Int where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized ([Tensor] -> Tensor) where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized [Layer]

data Darknet = Darknet [Layer]
  deriving (Generic)

instance Parameterized Darknet

darknetForward :: Darknet -> Tensor -> Tensor
darknetForward (Darknet layers) input = darknetForward' layers input [] []

darknetForward' :: [Layer] -> Tensor -> [Tensor] -> [Tensor] -> Tensor
darknetForward' [] input outputs yolo_outputs = cat (Dim 0) yolo_outputs
darknetForward' (x : xs) input outputs yolo_outputs =
  let (isYolo, output) =
        case x of
          Convolution {..} -> (False, func input)
          MaxPool {..} -> (False, func input)
          UpSample {..} -> (False, func input)
          Route {..} -> (False, route outputs)
          ShortCut {..} -> (False, shortcut outputs)
          Yolo {..} -> (True, func input)
   in darknetForward' xs output (output : outputs) (if isYolo then (output : yolo_outputs) else yolo_outputs)

instance Randomizable DarknetSpec Darknet where
  sample (DarknetSpec global layers) = do
    let previous_layers = reverse $ Nothing : map Just (reverse layers)
    Darknet <$> mapM (\(l, pl) -> sample' l pl) (zip layers previous_layers)
    where
      getSize :: Maybe LayerSpec -> Maybe Int
      getSize layer = size <$> layer
      sample' :: LayerSpec -> Maybe LayerSpec -> IO Layer
      sample' ConvolutionSpec {..} prev = do
        let outputChannelSize = filters
            inputChannelSize = fromMaybe (channels global) (getSize prev)
            kernelHeight = size
            kernelWidth = size
        weight <- makeIndependent =<< kaimingUniform FanIn (LeakyRelu $ Prelude.sqrt (5.0 :: Float)) [outputChannelSize, inputChannelSize, kernelHeight, kernelWidth]
        uniform <- randIO' [outputChannelSize]
        let fan_in = fromIntegral (kernelHeight * kernelWidth) :: Float
            bound = Prelude.sqrt $ (1 :: Float) / fan_in
            pad = (stride - 1) `div` 2
        bias <- makeIndependent =<< pure (mulScalar ((2 :: Float) * bound) (subScalar (0.5 :: Float) uniform))
        let func0 = conv2d' (toDependent weight) (toDependent bias) (stride, stride) (pad, pad)
            func1 input =
              if batch_normalize
                then I.batch_norm
                       (func0 input)
                       self.weight
                       _2
                       _1
                       _0
                       False
                       0.90000000000000002
                       1.0000000000000001e-05
                       True
                else func0 input
            func  = if 
        return $ Convolution {..}
      sample' MaxPoolSpec {..} _ = do
        let func = maxPool2d (size, size) (stride, stride) (pad, pad) (1, 1) False
            pad = (stride - 1) `div` 2
        return MaxPool {..}
      sample' UpSampleSpec {..} _ = do
        let func input = I.upsample_nearest2d input (stride, stride)
        return UpSample {..}
      sample' RouteSpec {..} _ = do
        let route inputs =
              let rinputs = reverse inputs
                  input i = if i < 0 then inputs !! (-1) else rinputs !! i
               in Prelude.foldl1 (+) $ map input layers
        return Route {..}
      sample' ShortCutSpec {..} _ = do
        let shortcut inputs =
              let rinputs = reverse inputs
                  input i = if i < 0 then inputs !! (-1) else rinputs !! i
               in input from
        return ShortCut {..}
      sample' YoloSpec {..} _ = do
        let func = id
        return Yolo {..}

parseIntList :: T.Text -> Either String [Int]
parseIntList line = listWithSeparator "," number line

configParser' :: IniParser [Either GlobalSpec LayerSpec]
configParser' = (toList <$>) $ sectionsOf pure $ \section -> do
  case section of
    "net" ->
      (Left <$>) $
        GlobalSpec
          <$> fieldOf "channels" number
          <*> fieldOf "height" number
    "convolution" ->
      (Right <$>) $
        ConvolutionSpec
          <$> fieldOf "batch_normalize" flag
          <*> fieldOf "filters" number
          <*> fieldOf "size" number
          <*> fieldOf "stride" number
          <*> fieldOf "activation" string
    "maxpool" ->
      (Right <$>) $
        MaxPoolSpec
          <$> fieldOf "size" number
          <*> fieldOf "stride" number
    "upsample" ->
      (Right <$>) $
        UpSampleSpec
          <$> fieldOf "stride" number
    "route" ->
      (Right <$>) $
        RouteSpec
          <$> fieldOf "layers" parseIntList
    "shortcut" ->
      (Right <$>) $
        ShortCutSpec
          <$> fieldOf "from" number
    "yolo" ->
      (Right <$>) $
        YoloSpec
          <$> fieldOf "mask" parseIntList
          <*> fieldOf "anchors" parseIntList
          <*> fieldOf "classes" number

readIniFile :: String -> IO (Either String DarknetSpec)
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
            else return $ Right $ DarknetSpec (head globalconfigs) configs
