{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveAnyClass #-}

module Torch.Typed.NN.Dropout where

import GHC.Generics
import System.IO.Unsafe
import Torch.NN (HasForward (..), Randomizable (..))
import Torch.Typed.Functional
import Torch.Typed.Tensor
import Torch.Typed.Parameter

data DropoutSpec where
  DropoutSpec ::
    {dropoutProbSpec :: Double} ->
    DropoutSpec
  deriving (Show, Eq)

data Dropout where
  Dropout ::
    {dropoutProb :: Double} ->
    Dropout
  deriving (Show, Generic, Parameterized)

dropoutForward ::
  forall shape dtype device.
  Dropout ->
  Bool ->
  Tensor device dtype shape ->
  IO (Tensor device dtype shape)
dropoutForward Dropout {..} dropoutTrain = dropout dropoutProb dropoutTrain

instance HasForward Dropout (Tensor device dtype shape) (Tensor device dtype shape) where
  forward dropout input = unsafePerformIO $ dropoutForward dropout False input
  forwardStoch dropout input = dropoutForward dropout True input

instance Randomizable DropoutSpec Dropout where
  sample DropoutSpec {..} = return $ Dropout dropoutProbSpec
