{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.NN.Dropout where

import           Control.Monad.State.Strict
import           Torch.HList
import           Data.Kind                    (Type)
import           Data.Proxy
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           GHC.Generics
import           System.IO.Unsafe

import qualified Torch.NN                      as A
import           Torch.NN                     (HasForward(..))
import qualified Torch.Autograd                as A
import qualified Torch.Tensor                  as A
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Functional
import           Torch.Typed.Tensor
import           Torch.Typed.Parameter
import           Torch.Typed.Device

data DropoutSpec
 where
  DropoutSpec
    :: { dropoutProbSpec :: Double }
    -> DropoutSpec
 deriving (Show, Eq)

data Dropout where
  Dropout
    :: { dropoutProb :: Double }
    -> Dropout
 deriving (Show, Generic)

dropoutForward
  :: forall shape dtype device
   . Dropout
  -> Bool
  -> Tensor device dtype shape
  -> IO (Tensor device dtype shape)
dropoutForward Dropout {..} dropoutTrain = dropout dropoutProb dropoutTrain

instance HasForward Dropout (Tensor device dtype shape) (Tensor device dtype shape) where
  forward dropout input = unsafePerformIO $ dropoutForward dropout False input
  forwardStoch dropout input = dropoutForward dropout True input

instance A.Randomizable DropoutSpec Dropout where
  sample DropoutSpec {..} = return $ Dropout dropoutProbSpec 
