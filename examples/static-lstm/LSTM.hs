{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedLists #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module LSTM where

import           Prelude                 hiding ( tanh )
import           Control.Monad                  ( foldM
                                                , when
                                                )
import           Data.List                      ( foldl'
                                                , scanl'
                                                , intersperse
                                                )
import           Data.Reflection
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra

import           Torch.Static
import           Torch.Static.Native     hiding ( linear )
import           Torch.Static.Factories
import qualified Torch.Autograd                as A
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Functions               as D
import qualified Torch.TensorFactories         as D

--------------------------------------------------------------------------------
-- Multi-Layer Perceptron (MLP)
--------------------------------------------------------------------------------


newtype Parameter dtype shape = Parameter A.IndependentTensor deriving (Show)

toDependent :: Parameter dtype shape -> Tensor dtype shape
toDependent (Parameter t) = UnsafeMkTensor $ A.toDependent t

instance A.Parameterized (Parameter dtype shape) where
    flattenParameters (Parameter x) = [x]
    replaceOwnParameters _ = Parameter <$> A.nextParameter


data LinearSpec (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) = LinearSpec
    deriving (Show, Eq)

data Linear (dtype :: D.DType) (inputFeatures :: Nat) (outputFeatures :: Nat) =
    Linear { weight :: Parameter dtype '[inputFeatures, outputFeatures]
        , bias :: Parameter dtype '[outputFeatures]
        } deriving (Show, Generic)



linear :: forall dtype (inputFeatures :: Nat) (outputFeatures :: Nat) (shape :: [Nat]) (shape' :: [Nat])
    . ( CheckBroadcast (CheckMatMul
                        shape
                        '[inputFeatures, outputFeatures]
                        (ComputeMatMul
                            (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                    '[outputFeatures]
                    (ComputeBroadcast
                        (ReverseImpl
                            (CheckMatMul
                                shape
                                '[inputFeatures, outputFeatures]
                                (ComputeMatMul
                                (ReverseImpl shape '[]) '[outputFeatures, inputFeatures]))
                            '[])
                        '[outputFeatures])
                    ~ shape')
    => Linear dtype inputFeatures outputFeatures
    -> Tensor dtype shape
    -> Tensor dtype shape'
linear Linear {..} input =
    add (matmul input (toDependent weight)) (toDependent bias)
      

data LSTMCell (inputDim :: Nat) (hiddenDim :: Nat) =  LSTMCell {
    lSTMCell_w_ih :: Parameter Double '[4 * hiddenDim, inputDim]  -- ^ input-to-hidden weights 
    , lSTMCell_w_hh :: Parameter Double '[4 * hiddenDim, hiddenDim] -- ^ hidden-to-hidden weights 
    , lSTMCell_b_ih :: Parameter Double '[4 * hiddenDim] -- ^ input-to-hidden bias
    , lSTMCell_b_hh :: Parameter Double '[4 * hiddenDim] -- ^ hidden-to-hidden bias 
    } deriving Generic
