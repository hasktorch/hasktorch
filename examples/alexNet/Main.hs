{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where
    
import GHC.Generics

import Torch.Serialize
import Torch.NN
import Torch.Functional
import Torch.TensorFactories
import Torch.TensorOptions
import Torch.Tensor
import qualified Torch.Functional.Internal as I

data ANSpec = ANSpec { 
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
    ANSpec 
        (Conv2dSpec 3 64 11 11 ) 
        (Conv2dSpec 64 192 5 5 ) 
        (Conv2dSpec 192 384 3 3 ) 
        (Conv2dSpec 384 256 3 3 ) 
        (Conv2dSpec 256 256 3 3 ) 
        (LinearSpec (6*6*256) 4096) 
        (LinearSpec 4096 4096) 
        (LinearSpec 4096 1000)

data AN = AN { 
    c1 :: Conv2d,
    c2 :: Conv2d,
    c3 :: Conv2d,
    c4 :: Conv2d,
    c5 :: Conv2d,
    l1 :: Linear,
    l2 :: Linear,
    l3 :: Linear
    } deriving (Generic, Show)

instance Parameterized AN
instance Randomizable ANSpec AN where
    sample ANSpec {..} = AN 
        <$> sample (conv1)
        <*> sample (conv2)
        <*> sample (conv3)
        <*> sample (conv4)
        <*> sample (conv5)
        <*> sample (fc1)
        <*> sample (fc2)
        <*> sample (fc3)

an :: AN -> Tensor -> Tensor
an AN{..} input = 
    linear l3
    . relu
    . linear l2
    . relu
    . linear l1
    . I.flatten 1 (-1) 
    . adaptiveAvgPool2d (6, 6)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) False
    . relu
    . conv2d'' c5 (1, 1) (1, 1)
    . relu
    . conv2d'' c4 (1, 1) (1, 1)
    . relu
    . conv2d'' c3 (1, 1) (1, 1)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) False
    . relu
    . conv2d'' c2 (1, 1) (2, 2)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) False
    . relu
    . conv2d'' c1 (4, 4) (2, 2)
    $ input

main :: IO ()
main = do
    model <- sample spec
    model <-  loadParams model "alexNet.pt"
    let input = ones [1, 3, 224, 224] defaultOpts
    print $ an model input
