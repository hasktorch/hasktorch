{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}

module Main where

import           Data.Proxy
import           GHC.TypeLits
import           GHC.Generics

import qualified Torch.NN                      as A
import qualified Torch.Autograd                as A
import qualified Torch.Tensor                  as A
import qualified Torch.DType                   as D
import           Torch.Static
import           Torch.Static.Native
import           Torch.Static.NN

type NoStrides = '(1, 1)
type NoPadding = '(0, 0)

type KernelSize = '(2, 2)
type Strides = '(2, 2)

data CNNSpec (dtype :: D.DType)
  = CNNSpec deriving (Show, Eq)

data CNN (dtype :: D.DType)
 where
  CNN
    :: forall dtype
     . { conv0 :: Conv2d dtype 1  20 5 5
       , conv1 :: Conv2d dtype 20 50 5 5
       , fc0   :: Linear dtype (4*4*50) 500
       , fc1   :: Linear dtype 500      10
       }
    -> CNN dtype
 deriving (Show, Generic)

cnn
  :: forall dtype batchSize
   . _
  => CNN dtype
  -> Tensor dtype '[batchSize, 1, 28, 28]
  -> Tensor dtype '[batchSize, 10]
cnn CNN {..} =
  logSoftmax @1
    . Torch.Static.NN.linear fc1
    . relu
    . Torch.Static.NN.linear fc0
    . reshape @'[batchSize, 4*4*50]
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . Torch.Static.NN.conv2d @NoStrides @NoPadding conv1
    . maxPool2d @KernelSize @Strides @NoPadding
    . relu
    . Torch.Static.NN.conv2d @NoStrides @NoPadding conv0

instance A.Parameterized (CNN dtype)

instance (KnownDType dtype)
  => A.Randomizable (CNNSpec dtype)
                    (CNN     dtype)
 where
  sample CNNSpec =
    CNN
      <$> A.sample (Conv2dSpec @dtype @1  @20 @5 @5)
      <*> A.sample (Conv2dSpec @dtype @20 @50 @5 @5)
      <*> A.sample (LinearSpec @dtype @(4*4*50) @500)
      <*> A.sample (LinearSpec @dtype @500      @10)

main = undefined
