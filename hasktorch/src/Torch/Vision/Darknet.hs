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
import Torch.Functional as D
import qualified Torch.Functional.Internal as I
import Torch.Initializers
import Torch.NN
import Torch.Tensor as D
import Torch.TensorFactories

data GlobalSpec
  = GlobalSpec
      { channels :: Int,
        height :: Int,
        train :: Bool
      }
  deriving (Show, Eq)

data LayerSpec
  = ConvolutionSpec
      { batch_normalize :: Bool,
        filters :: Int,
        layer_size :: Int,
        stride :: Int,
        activation :: String
      }
  | MaxPoolSpec
      { layer_size :: Int,
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
        anchors :: [(Int, Int)],
        classes :: Int
      }
  deriving (Show, Eq)

data DarknetSpec = DarknetSpec GlobalSpec [LayerSpec]
  deriving (Show)

data Layer
  = Convolution
      { weight :: Parameter,
        bias :: Parameter,
        batchNormWeight :: Maybe Parameter,
        batchNormBias :: Maybe Parameter,
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
      { yolo :: Tensor -> YoloOutput
      }
  deriving (Generic)

type YoloOutput = (Tensor,Tensor)

instance Parameterized Layer

instance Parameterized (Maybe Parameter)

instance Parameterized Int where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized ([Tensor] -> Tensor) where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized (Tensor -> YoloOutput) where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized [Layer]

data Darknet = Darknet [Layer]
  deriving (Generic)

instance Parameterized Darknet

darknetForward :: Darknet -> Tensor -> YoloOutput
darknetForward (Darknet layers) input = darknetForward' layers input [] []

darknetForward' :: [Layer] -> Tensor -> [Tensor] -> [YoloOutput] -> YoloOutput
darknetForward' [] input outputs yolo_outputs = (cat (Dim 0) (map fst yolo_outputs), cat (Dim 0) (map snd yolo_outputs))
darknetForward' (x : xs) input outputs yolo_outputs =
  let moutput =
        case x of
          Convolution {..} -> Just $ func input
          MaxPool {..} -> Just $ func input
          UpSample {..} -> Just $ func input
          Route {..} -> Just $ route outputs
          ShortCut {..} -> Just $ shortcut outputs
          Yolo {..} -> Nothing
   in
    case moutput of
      Just output -> darknetForward' xs output (output : outputs) yolo_outputs
      Nothing ->
        let yolo_output = (yolo x) input
        in darknetForward' xs input (input : outputs) (yolo_output : yolo_outputs)

instance Randomizable DarknetSpec Darknet where
  sample (DarknetSpec global layers) = do
    let previous_layers = reverse $ Nothing : map Just (reverse layers)
    Darknet <$> mapM (\(l, pl) -> sample' l pl) (zip layers previous_layers)
    where
      getSize :: Maybe LayerSpec -> Maybe Int
      getSize layer = layer_size <$> layer
      sample' :: LayerSpec -> Maybe LayerSpec -> IO Layer
      sample' ConvolutionSpec {..} prev = do
        -- convolution 2d --
        let outputChannelSize = filters
            inputChannelSize = fromMaybe (channels global) (getSize prev)
            kernelHeight = layer_size
            kernelWidth = layer_size
        weight <- makeIndependent =<< kaimingUniform FanIn (LeakyRelu $ Prelude.sqrt (5.0 :: Float)) [outputChannelSize, inputChannelSize, kernelHeight, kernelWidth]
        uniform <- randIO' [outputChannelSize]
        let fan_in = fromIntegral (kernelHeight * kernelWidth) :: Float
            bound = Prelude.sqrt $ (1 :: Float) / fan_in
            pad = (stride - 1) `div` 2
        bias <- makeIndependent =<< pure (mulScalar ((2 :: Float) * bound) (subScalar (0.5 :: Float) uniform))
        let func0 = conv2d' (toDependent weight) (toDependent bias) (stride, stride) (pad, pad)
        -- batchnorm 2d --
        ( batchNormWeight,
          batchNormBias,
          func1
          ) <-
          do
            if batch_normalize
              then do
                batchNormWeight' <- makeIndependent (ones' [outputChannelSize])
                batchNormBias' <- makeIndependent (zeros' [outputChannelSize])
                runningMean <- makeIndependent (zeros' [outputChannelSize])
                runningVar <- makeIndependent (ones' [outputChannelSize])
                return
                  ( Just batchNormWeight',
                    Just batchNormBias',
                    \input ->
                      I.batch_norm
                        (func0 input)
                        (toDependent batchNormWeight')
                        (toDependent batchNormBias')
                        (toDependent runningMean)
                        (toDependent runningVar)
                        (train global)
                        0.90000000000000002
                        1.0000000000000001e-05
                        True
                  )
              else do
                return
                  ( Nothing,
                    Nothing,
                    func0
                  )
        -- leaky relu --
        let func input = if activation == "leaky" then I.leaky_relu input 0.1 else func1 input
        return Convolution {..}
      sample' MaxPoolSpec {..} _ = do
        let func = maxPool2d (layer_size, layer_size) (stride, stride) (pad, pad) (1, 1) False
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
        let yolo :: Tensor -> (Tensor,Tensor)
            yolo input =
              let num_samples = D.size input 0
                  grid_size = D.size input 2
                  g = grid_size
                  num_anchors = (length anchors) * 2
                  img_dim = shape input !! 2
                  stride = (fromIntegral img_dim) / (fromIntegral grid_size) :: Float
                  prediction = contiguous $ permute [0, 1, 3, 4, 2] $ view [num_samples, num_anchors, classes + 5, grid_size, grid_size] input
                  x = sigmoid $ slice 0 0 1 1 $ prediction
                  y = sigmoid $ slice 0 1 2 1 $ prediction
                  w = slice 0 2 3 1 $ prediction
                  h = slice 0 3 4 1 $ prediction
                  pred_conf = sigmoid $ slice 0 4 5 1 $ prediction
                  pred_cls = sigmoid $ slice 0 5 (-1) 1 $ prediction
                  grid_x = view [1, 1, g, g] $ D.repeat [g, 1] $ arange' (0 :: Int) g (1 :: Int)
                  grid_y = view [1, 1, g, g] $ I.t $ D.repeat [g, 1] $ arange' (0 :: Int) g (1 :: Int)
                  scaled_anchors = map (\(a_w, a_h) -> ((fromIntegral a_w) / stride, (fromIntegral a_h) / stride)) anchors
                  anchor_w = view [1, num_anchors, 1, 1] $ asTensor $ (map fst scaled_anchors  :: [Float])
                  anchor_h = view [1, num_anchors, 1, 1] $ asTensor $ (map snd scaled_anchors :: [Float])
                  pred_boxes =
                    cat
                      (Dim 3)
                      [ x + grid_x,
                        y + grid_y,
                        D.exp w * anchor_w,
                        D.exp h * anchor_h
                      ]
                  output =
                    cat
                      (Dim (-1))
                      [ stride `mulScalar` view [num_samples, -1, 4] pred_boxes,
                        view [num_samples, -1, 1] pred_conf,
                        view [num_samples, -1, classes] pred_cls
                      ]
                  -- loss_x = mseLoss(x[obj_mask], tx[obj_mask])
                  -- loss_y = mseLoss(y[obj_mask], ty[obj_mask])
                  -- loss_w = mseLoss(w[obj_mask], tw[obj_mask])
                  -- loss_h = mseLoss(h[obj_mask], th[obj_mask])
                  -- loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
                  -- loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
                  -- loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                  -- loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
                  -- total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
                  total_loss = undefined
               in (output, total_loss)
        return Yolo {..}

parseIntList :: T.Text -> Either String [Int]
parseIntList line = listWithSeparator "," number line

parseInt2List :: T.Text -> Either String [(Int, Int)]
parseInt2List line = do
  intlist <- listWithSeparator "," number line
  let loop [] _ = []
      loop (x : xs) Nothing = loop xs (Just x)
      loop (y : xs) (Just x) = (x, y) : loop xs Nothing
  return $ loop intlist Nothing

configParser' :: IniParser [Either GlobalSpec LayerSpec]
configParser' = (toList <$>) $ sectionsOf pure $ \section -> do
  case section of
    "net" ->
      (Left <$>) $
        GlobalSpec
          <$> fieldOf "channels" number
          <*> fieldOf "height" number
          <*> pure True
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
          <*> fieldOf "anchors" parseInt2List
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
