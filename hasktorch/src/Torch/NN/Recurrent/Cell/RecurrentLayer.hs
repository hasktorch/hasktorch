{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.RecurrentLayer where

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Torch
