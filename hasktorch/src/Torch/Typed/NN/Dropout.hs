{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Typed.NN.Dropout where

import GHC.Generics
import System.IO.Unsafe
import Torch.NN (HasForward (..), Randomizable (..))
import Torch.Typed.Functional
import Torch.Typed.Parameter
import Torch.Typed.Tensor

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

instance HasForward Dropout (Tensor device dtype shape) where
  type Output Dropout (Tensor device dtype shape) = Tensor device dtype shape
  forward dropout input = unsafePerformIO $ dropoutForward dropout False input

instance Randomizable DropoutSpec Dropout where
  sample DropoutSpec {..} = return $ Dropout dropoutProbSpec
