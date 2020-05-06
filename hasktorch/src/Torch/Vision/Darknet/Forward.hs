{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Vision.Darknet.Forward where

import Control.Monad (forM, mapM)
import Data.List ((!!))
import Data.Map ((!), Map, empty, insert)
import Data.Maybe (isJust)
import GHC.Exts
import GHC.Generics
import Torch.Autograd
import qualified Torch.Functional as D
import qualified Torch.Functional.Internal as I
import Torch.NN
import Torch.Tensor as D
import Torch.DType as D
import Torch.TensorFactories
import Torch.Typed.NN (HasForward (..))
import qualified Torch.Vision.Darknet.Spec as S

type Index = Int

type Loss = Tensor

data ConvolutionWithBatchNorm
  = ConvolutionWithBatchNorm
      { conv2d :: Conv2d,
        batchNorm :: BatchNorm,
        stride :: Int,
        layerSize :: Int,
        isLeaky :: Bool
      }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.ConvolutionWithBatchNormSpec ConvolutionWithBatchNorm where
  sample S.ConvolutionWithBatchNormSpec {..} = do
    ConvolutionWithBatchNorm
      <$> sample
        ( Conv2dSpec
            { inputChannelSize = input_filters,
              outputChannelSize = filters,
              kernelHeight = layer_size,
              kernelWidth = layer_size
            }
        )
      <*> sample
        ( BatchNormSpec
            { numFeatures = filters
            }
        )
      <*> pure stride
      <*> pure layer_size
      <*> pure (activation == "leaky")

instance HasForward ConvolutionWithBatchNorm (Bool, Tensor) Tensor where
  forward ConvolutionWithBatchNorm {..} (train, input) =
    let pad = (layerSize - 1) `div` 2
        activation = if isLeaky then flip I.leaky_relu 0.1 else id
     in activation
          $ batchNormForward batchNorm train 0.90000000000000002 1.0000000000000001e-05
          $ conv2dForward conv2d (stride, stride) (pad, pad) input

data Convolution
  = Convolution
      { conv2d :: Conv2d,
        stride :: Int,
        layerSize :: Int,
        isLeaky :: Bool
      }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.ConvolutionSpec Convolution where
  sample S.ConvolutionSpec {..} = do
    Convolution
      <$> sample
        ( Conv2dSpec
            { inputChannelSize = input_filters,
              outputChannelSize = filters,
              kernelHeight = layer_size,
              kernelWidth = layer_size
            }
        )
      <*> pure stride
      <*> pure layer_size
      <*> pure (activation == "leaky")

instance HasForward Convolution Tensor Tensor where
  forward Convolution {..} input =
    let pad = (layerSize - 1) `div` 2
     in conv2dForward conv2d (stride, stride) (pad, pad) input

data MaxPool
  = MaxPool
      { stride :: Int,
        layerSize :: Int
      }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.MaxPoolSpec MaxPool where
  sample S.MaxPoolSpec {..} = do
    MaxPool
      <$> pure stride
      <*> pure layer_size

instance HasForward MaxPool Tensor Tensor where
  forward MaxPool {..} input =
    let pad = (layerSize - 1) `div` 2
     in D.maxPool2d (layerSize, layerSize) (stride, stride) (pad, pad) (1, 1) False input

data UpSample
  = UpSample
      { stride :: Int
      }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.UpSampleSpec UpSample where
  sample S.UpSampleSpec {..} = do
    UpSample
      <$> pure stride

instance HasForward UpSample Tensor Tensor where
  forward UpSample {..} input =
    I.upsample_nearest2d input (stride, stride) (-1) (-1)

data Route
  = Route
      { layers :: [Int]
      }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.RouteSpec Route where
  sample S.RouteSpec {..} = do
    Route
      <$> pure layers

instance HasForward Route (Map Int Tensor) Tensor where
  forward Route {..} inputs =
    D.cat (D.Dim 1) (map (inputs !) layers)

data ShortCut
  = ShortCut
      { from :: Int
      }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.ShortCutSpec ShortCut where
  sample S.ShortCutSpec {..} = do
    ShortCut
      <$> pure from

instance HasForward ShortCut (Map Int Tensor) Tensor where
  forward ShortCut {..} inputs = inputs ! from

type Anchors = [(Float,Float)]
type ScaledAnchors = [(Float,Float)]

data Yolo
  = Yolo
      { anchors :: Anchors,
        classes :: Int,
        img_size :: Int
      }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.YoloSpec Yolo where
  sample S.YoloSpec {..} = pure $ Yolo { classes = classes ,
                                         anchors = map (\(a,b) -> (fromIntegral a,fromIntegral b))anchors,
                                         img_size = img_size
                                       
                                       }

newtype Prediction = Prediction {fromPrediction :: Tensor} deriving (Show)

toPrediction :: Yolo -> Tensor -> Prediction
toPrediction Yolo {..} input =
  let num_samples = D.size input 0
      grid_size = D.size input 2
      num_anchors = length anchors
   in Prediction $ D.contiguous $ D.permute [0, 1, 3, 4, 2] $ D.view [num_samples, num_anchors, classes + 5, grid_size, grid_size] input

squeezeLastDim :: Tensor -> Tensor
squeezeLastDim input = I.squeezeDim input (-1)

toX ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toX prediction = D.sigmoid $ squeezeLastDim (D.slice (-1) 0 1 1 $ fromPrediction prediction)

toY ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toY prediction = D.sigmoid $ squeezeLastDim (D.slice (-1) 1 2 1 $ fromPrediction prediction)

toW ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toW prediction = squeezeLastDim (D.slice (-1) 2 3 1 $ fromPrediction prediction)

toH ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toH prediction = squeezeLastDim (D.slice (-1) 3 4 1 $ fromPrediction prediction)

toPredConf ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toPredConf prediction = D.sigmoid $ squeezeLastDim (D.slice (-1) 4 5 1 $ fromPrediction prediction)

toPredClass ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid, class]
  Tensor
toPredClass prediction =
  let input = fromPrediction prediction
      num_features = D.size input (-1)
   in D.sigmoid $ squeezeLastDim (D.slice (-1) 5 num_features 1 input)

gridX ::
  -- |  grid size
  Int ->
  -- |  [1, 1, grid, grid]
  Tensor
gridX g = D.view [1, 1, g, g] $ D.repeat [g, 1] $ arange' (0 :: Int) g (1 :: Int)

gridY ::
  -- |  grid size
  Int ->
  -- |  [1, 1, grid, grid]
  Tensor
gridY g = D.contiguous $ D.view [1, 1, g, g] $ I.t $ D.repeat [g, 1] $ arange' (0 :: Int) g (1 :: Int)


toScaledAnchors :: Anchors -> Float -> ScaledAnchors
toScaledAnchors anchors stride =  map (\(a_w, a_h) -> (a_w / stride, a_h / stride)) anchors

toAnchorW :: ScaledAnchors -> Tensor
toAnchorW scaled_anchors = D.view [1, length scaled_anchors, 1, 1] $ asTensor $ (map fst scaled_anchors :: [Float])

toAnchorH :: ScaledAnchors -> Tensor
toAnchorH scaled_anchors = D.view [1, length scaled_anchors, 1, 1] $ asTensor $ (map snd scaled_anchors :: [Float])

toPredBox ::
  Yolo ->
  Prediction ->
  (Tensor,Tensor,Tensor,Tensor)
toPredBox Yolo {..} prediction =
  let input = fromPrediction prediction
      grid_size = D.size input 2
      stride = fromIntegral img_size / fromIntegral grid_size :: Float
      scaled_anchors = toScaledAnchors anchors stride
      anchor_w = toAnchorW scaled_anchors
      anchor_h = toAnchorH scaled_anchors
   in ( toX prediction + gridX grid_size,
        toY prediction + gridY grid_size,
        D.exp (toW prediction) * anchor_w,
        D.exp (toH prediction) * anchor_h
        )

bboxWhIou
  :: (Float,Float)
  -> (Tensor,Tensor) -- ^ (batch, batch)
  -> Tensor -- ^ batch
bboxWhIou (w1',h1') (w2,h2) =
  let w1 = asTensor w1'
      h1 = asTensor h1'
      inter_area = I.min w1 w2 * I.min h1 h2
      union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    in inter_area / union_area

data Target = Target
  { obj_mask :: Tensor
  , noobj_mask :: Tensor
  , tx :: Tensor
  , ty :: Tensor
  , tw :: Tensor
  , th :: Tensor
  , tcls :: Tensor
  , tconf :: Tensor
  }


toBuildTargets
  :: (Tensor,Tensor,Tensor,Tensor)
  -> Tensor
  -> Tensor
  -> Anchors
  -> Float
  -> Target
toBuildTargets (pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h) pred_cls target anchors ignore_thres =
  let nB = D.size pred_boxes_x 0
      nA = D.size pred_boxes_x 1
      nC = D.size pred_cls (-1)
      nG = D.size pred_boxes_x 2
      obj_mask_init = zeros [nB,nA,nG,nG] bool_opts
      noobj_mask_init = ones [nB,nA,nG,nG] bool_opts
      class_mask_init = zeros' [nB,nA,nG,nG]
      iou_scores_init = zeros' [nB,nA,nG,nG]
      tx_init = zeros' [nB,nA,nG,nG]
      ty_init = zeros' [nB,nA,nG,nG]
      tcls_init = zeros' [nB,nA,nG,nG,nC]
      target_boxes =  nG `D.mulScalar` (D.slice (-1) 2 6 1 target)
      gx = squeezeLastDim $  D.slice (-1) 0 1 1 target_boxes
      gy = squeezeLastDim $ D.slice (-1) 1 2 1 target_boxes
      gw =squeezeLastDim $  D.slice (-1) 2 3 1 target_boxes
      gh = squeezeLastDim $ D.slice (-1) 3 4 1 target_boxes
      gi = toType D.Int64 gx
      gj = toType D.Int64 gy
      -- (anchors,batch)
      ious_list = map (\anchor -> bboxWhIou anchor (gw,gh)) anchors
      ious = D.stack (D.Dim 0) ious_list
      (best_ious,best_n) = I.maxDim ious 0 False
      best_n_anchor = anchors !! (asValue best_n::Int)
      b = squeezeLastDim $ D.slice (-1) 0 1 1 target
      target_labels = squeezeLastDim $ D.slice (-1) 1 2 1 target
      obj_mask = indexPut
                   obj_mask_init
                   [b,best_n,gj,gi]
                   True
      noobj_mask' = indexPut
                     noobj_mask_init
                     [b,best_n,gj,gi]
                     False
      noobj_mask = indexPut
                    noobj_mask'
                    [b, ious `D.gt` (asTensor ignore_thres), gj, gi]
                    False
      tx = indexPut
             (zeros' [nB,nA,nG,nG])
             [b,best_n,gj,gi]
             (gx - D.floor gx)
      ty = indexPut
             (zeros' [nB,nA,nG,nG])
             [b,best_n,gj,gi]
             (gy - D.floor gy)
      tw = indexPut
             (zeros' [nB,nA,nG,nG])
             [b,best_n,gj,gi]
             (I.log (gw / (asTensor (fst best_n_anchor))+ 1e-16))
      th = indexPut
             (zeros' [nB,nA,nG,nG])
             [b,best_n,gj,gi]
             (I.log (gh / (asTensor (snd best_n_anchor))+ 1e-16))
      tcls = indexPut
               tcls_init
               [b, best_n, gj, gi, target_labels]
               (1::Float)
      tconf = toType D.Float obj_mask
  in Target {..}


index :: TensorLike a => Tensor -> [a] -> Tensor
index org idx = I.index org (map asTensor idx)

indexPut :: (TensorLike a, TensorLike b) => Tensor -> [a] -> b -> Tensor
indexPut org idx value = I.index_put org (map asTensor idx) (asTensor value) False

totalLoss :: Yolo -> Prediction -> Target -> Tensor
totalLoss yolo prediction Target {..} =
  let x = toX prediction
      y = toY prediction
      w = toW prediction
      h = toH prediction
      pred_conf = toPredConf prediction
      pred_cls = toPredClass prediction
      omask t = t `index` [obj_mask]
      nmask t = t `index` [noobj_mask]
      loss_x = D.mseLoss (omask x) (omask ty)
      loss_y = D.mseLoss (omask y) (omask ty)
      loss_w = D.mseLoss (omask w) (omask tw)
      loss_h = D.mseLoss (omask h) (omask th)
      bceLoss = D.binaryCrossEntropyLoss'
      loss_conf_obj = bceLoss (omask pred_conf) (omask tconf)
      loss_conf_noobj = bceLoss (nmask pred_conf) (nmask tconf)
      obj_scale = 1
      noobj_scale = 100
      loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
      loss_cls = bceLoss (omask pred_cls) (omask tcls)
  in loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls


data YoloOutput
  = YoloOutput
      { x :: Tensor,
        y :: Tensor,
        w :: Tensor,
        h :: Tensor,
        predBoxes :: Tensor,
        predConf :: Tensor,
        predClass :: Tensor
      }
  deriving (Show)


instance HasForward Yolo (Maybe Tensor, Tensor) Tensor where
  forward yolo@Yolo {..} (train, input) =
    let num_samples = D.size input 0
        grid_size = D.size input 2
        g = grid_size
        num_anchors = length anchors
        img_dim = shape input !! 2
        stride = (fromIntegral img_dim) / (fromIntegral grid_size) :: Float
        prediction = toPrediction yolo input
        pred_boxes = toPredBox yolo prediction
        (px,py,pw,ph) = pred_boxes
        pred_cls = toPredClass prediction
        pred_conf = toPredConf prediction
     in case train of
        Nothing -> ( D.cat
               (D.Dim (-1))
               [ stride `D.mulScalar` D.view [num_samples, -1, 4] (D.cat (D.Dim 3) [px,py,pw,ph]),
                 D.view [num_samples, -1, 1] pred_conf,
                 D.view [num_samples, -1, classes] pred_cls
               ]
             )
        Just target ->
          let ignore_thres = 0.5
              build_target = toBuildTargets pred_boxes pred_cls target anchors ignore_thres
          in totalLoss yolo prediction build_target


data Layer
  = LConvolution Convolution
  | LConvolutionWithBatchNorm ConvolutionWithBatchNorm
  | LMaxPool MaxPool
  | LUpSample UpSample
  | LRoute Route
  | LShortCut ShortCut
  | LYolo Yolo
  deriving (Show, Generic, Parameterized)

data Darknet = Darknet [(Index, Layer)] deriving (Show, Generic, Parameterized)

instance Randomizable S.DarknetSpec Darknet where
  sample (S.DarknetSpec layers) = do
    layers <- forM (toList layers) $ \(idx, layer) ->
      case layer of
        S.LConvolutionSpec s -> (\s -> (idx, (LConvolution s))) <$> sample s
        S.LConvolutionWithBatchNormSpec s -> (\s -> (idx, (LConvolutionWithBatchNorm s))) <$> sample s
        S.LMaxPoolSpec s -> (\s -> (idx, (LMaxPool s))) <$> sample s
        S.LUpSampleSpec s -> (\s -> (idx, (LUpSample s))) <$> sample s
        S.LRouteSpec s -> (\s -> (idx, (LRoute s))) <$> sample s
        S.LShortCutSpec s -> (\s -> (idx, (LShortCut s))) <$> sample s
        S.LYoloSpec s -> (\s -> (idx, (LYolo s))) <$> sample s
    pure $ Darknet (fromList layers)

instance HasForward Darknet (Maybe Tensor, Tensor) Tensor where
  forward (Darknet layers) (train, input) =
    let loop :: [(Index, Layer)] -> (Map Index Tensor) -> [Tensor] -> Tensor
        loop [] _ tensors = D.cat (D.Dim 1) tensors
        loop ((idx, layer) : next) layerOutputs yoloOutputs =
          let input' = (if idx == 0 then input else layerOutputs ! (idx -1))
           in case layer of
                LConvolution s ->
                  let out = forward s input'
                   in loop next (insert idx out layerOutputs) yoloOutputs
                LConvolutionWithBatchNorm s ->
                  let out = forward s (isJust train, input')
                   in loop next (insert idx out layerOutputs) yoloOutputs
                LMaxPool s ->
                  let out = forward s input'
                   in loop next (insert idx out layerOutputs) yoloOutputs
                LUpSample s ->
                  let out = forward s input'
                   in loop next (insert idx out layerOutputs) yoloOutputs
                LRoute s ->
                  let out = forward s layerOutputs
                   in loop next (insert idx out layerOutputs) yoloOutputs
                LShortCut s ->
                  let out = forward s layerOutputs
                   in loop next (insert idx out layerOutputs) yoloOutputs
                LYolo s ->
                  let out = forward s (train, input')
                   in loop next layerOutputs (out : yoloOutputs)
     in loop layers empty []
