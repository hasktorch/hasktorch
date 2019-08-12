{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DeriveGeneric #-}

module Torch.NN where

import Control.Monad.State.Strict

import Torch.Autograd
import Torch.Tensor
import Torch.TensorFactories (ones', rand', randn')
import Torch.Functions
import GHC.Generics

class Randomizable spec f | spec -> f where
    sample :: spec -> IO f
    default sample :: (Generic spec, Randomizable' (Rep spec)) => spec -> IO f
    sample spec = sample' (from spec) 