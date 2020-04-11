{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where
    
import GHC.Generics

import Torch.Serialize
import Torch.Vision
import Torch.NN
import Torch.Functional
import Torch.TensorFactories
import Torch.TensorOptions
import Torch.Tensor
import qualified Torch.DType as D

data AlexNetSpec = AlexNetSpec { 
    conv1 :: Conv2dSpec,         
    conv2 :: Conv2dSpec,         
    conv3 :: Conv2dSpec,         
    conv4 :: Conv2dSpec,
    conv5 :: Conv2dSpec,
    fc1   :: LinearSpec,              
    fc2   :: LinearSpec,
    fc3   :: LinearSpec
    } deriving (Show, Eq)

spec = 
    AlexNetSpec 
        (Conv2dSpec 3 64 11 11 ) 
        (Conv2dSpec 64 192 5 5 ) 
        (Conv2dSpec 192 384 3 3 ) 
        (Conv2dSpec 384 256 3 3 ) 
        (Conv2dSpec 256 256 3 3 ) 
        (LinearSpec (6*6*256) 4096) 
        (LinearSpec 4096 4096) 
        (LinearSpec 4096 1000)

data AlexNet = AlexNet { 
    c1 :: Conv2d,
    c2 :: Conv2d,
    c3 :: Conv2d,
    c4 :: Conv2d,
    c5 :: Conv2d,
    l1 :: Linear,
    l2 :: Linear,
    l3 :: Linear
    } deriving (Generic, Show)

instance Parameterized AlexNet
instance Randomizable AlexNetSpec AlexNet where
    sample AlexNetSpec {..} = AlexNet 
        <$> sample (conv1)
        <*> sample (conv2)
        <*> sample (conv3)
        <*> sample (conv4)
        <*> sample (conv5)
        <*> sample (fc1)
        <*> sample (fc2)
        <*> sample (fc3)

alexnetForward :: AlexNet -> Tensor -> Tensor
alexnetForward AlexNet{..} input = 
    linear l3
    . relu
    . linear l2
    . relu
    . linear l1
    . flatten 1 (-1) 
    . adaptiveAvgPool2d (6, 6)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) False
    . relu
    . conv2dForward c5 (1, 1) (1, 1)
    . relu
    . conv2dForward c4 (1, 1) (1, 1)
    . relu
    . conv2dForward c3 (1, 1) (1, 1)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) False
    . relu
    . conv2dForward c2 (1, 1) (2, 2)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) False
    . relu
    . conv2dForward c1 (4, 4) (2, 2)
    $ input

normalize :: Either String Tensor -> Tensor
normalize (Right img) = cat 0 [r', g', b']
    where
        img' = divScalar (255.0::Float) $ toType D.Float (hwc2chw img)
        [r, g, b] = I.split (reshape [3, 224, 224] img') 1 0 
        r' = divScalar (0.229::Float) (subScalar(0.485::Float) r) 
        g' = divScalar (0.224::Float) (subScalar(0.456::Float) g) 
        b' = divScalar (0.225::Float) (subScalar(0.406::Float) b) 

main :: IO ()
main = do
    model <- sample spec
    model <- loadParams model $ "alexNet/" ++ "alexNet.pt"
    putStrLn "Enter image name with file extension:"
    imgName <- getLine
    img <- readImage $ "alexNet/" ++ imgName
    let unNormalizedScores =  alexnetForward model $ reshape [1, 3, 224, 224] $ normalize img
        predictedLabelIndex = fromIntegral $ toInt $ argmax (Dim 1) KeepDim $ softmax 1 unNormalizedScores
    print predictedLabelIndex
