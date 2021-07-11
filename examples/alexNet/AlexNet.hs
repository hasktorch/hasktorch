{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module AlexNet where

import GHC.Generics
import Torch.Functional
import qualified Torch.Functional.Internal as I
import Torch.NN
import Torch.Tensor

data AlexNetBBSpec = AlexNetBBSpec
  { conv1 :: Conv2dSpec,
    conv2 :: Conv2dSpec,
    conv3 :: Conv2dSpec,
    conv4 :: Conv2dSpec,
    conv5 :: Conv2dSpec,
    fc1 :: LinearSpec,
    fc2 :: LinearSpec
  }
  deriving (Show, Eq)

alexnetBackBoneSpec =
  AlexNetBBSpec
    (Conv2dSpec 3 64 11 11)
    (Conv2dSpec 64 192 5 5)
    (Conv2dSpec 192 384 3 3)
    (Conv2dSpec 384 256 3 3)
    (Conv2dSpec 256 256 3 3)
    (LinearSpec (6 * 6 * 256) 4096)
    (LinearSpec 4096 4096)

data AlexNetBB = AlexNetBB
  { c1 :: Conv2d,
    c2 :: Conv2d,
    c3 :: Conv2d,
    c4 :: Conv2d,
    c5 :: Conv2d,
    l1 :: Linear,
    l2 :: Linear
  }
  deriving (Generic, Show)

instance Parameterized AlexNetBB

instance Randomizable AlexNetBBSpec AlexNetBB where
  sample AlexNetBBSpec {..} =
    AlexNetBB
      <$> sample (conv1)
      <*> sample (conv2)
      <*> sample (conv3)
      <*> sample (conv4)
      <*> sample (conv5)
      <*> sample (fc1)
      <*> sample (fc2)

alexnetBBForward :: AlexNetBB -> Tensor -> Tensor
alexnetBBForward AlexNetBB {..} input =
  linear l2
    . relu
    . linear l1
    . flatten (Dim 1) (Dim (-1))
    . adaptiveAvgPool2d (6, 6)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) Floor
    . relu
    . conv2dForward c5 (1, 1) (1, 1)
    . relu
    . conv2dForward c4 (1, 1) (1, 1)
    . relu
    . conv2dForward c3 (1, 1) (1, 1)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) Floor
    . relu
    . conv2dForward c2 (1, 1) (2, 2)
    . maxPool2d (3, 3) (2, 2) (0, 0) (1, 1) Floor
    . relu
    . conv2dForward c1 (4, 4) (2, 2)
    $ input
