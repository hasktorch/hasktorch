{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
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

module Main where

import GHC.TypeLits
import Data.Proxy

import qualified Torch.Tensor as D
import qualified Torch.Autograd as A
import qualified Torch.DType as DType
import Torch.Static

--------------------------------------------------------------------------------

data Conv2d dtype (in_features :: Nat) (out_features :: Nat)
                  (kernel_size :: (Nat, Nat))
                  (stride :: (Nat, Nat))
                  (padding :: (Nat, Nat)) =
    Conv2d { weight :: Tensor dtype '[out_features, in_features, Fst kernel_size, Snd kernel_size]
           , bias   :: Tensor dtype '[out_features] }


-- The constraints on this one are _very_ involved, so the partial signatures
-- make the code significantly cleaner.
conv2d :: forall stride padding.
        _ => Conv2d _ _ _ _ stride padding -> Tensor _ _ -> Tensor _ _
conv2d Conv2d{..} input = conv2dBias @stride @padding input weight bias

--------------------------------------------------------------------------------

data Linear dtype (in_features :: Nat) (out_features :: Nat) =
    Linear { weight :: Tensor dtype '[in_features, out_features]
           , bias   :: Tensor dtype '[out_features]
           }

linear :: Linear dtype in_features out_features ->
          Tensor dtype [n, in_features] ->
          Tensor dtype [n, out_features]
linear Linear{..} input = add (mm input weight) bias

--------------------------------------------------------------------------------

type NoPadding = '(0, 0)
type NoStrides = '(1, 1)

data Model dtype = Model { conv1 :: Conv2d dtype 1   20 '(5, 5) NoStrides NoPadding
                         , conv2 :: Conv2d dtype 20  50 '(5, 5) NoStrides NoPadding
                         , fc1   :: Linear dtype (4*4*50) 500
                         , fc2   :: Linear dtype 500      10
                         }

model :: forall dtype n. _ => Model dtype -> Tensor dtype [n, 1, 28, 28] -> Tensor dtype [n, 10]
model Model{..} x = output
  where
    c1     = relu $ conv2d conv1 x
    p1     = maxPool2d @'(2, 2) @'(2, 2) @NoPadding c1
    c2     = relu $ conv2d conv2 p1
    p2     = maxPool2d @'(2, 2) @'(2, 2) @NoPadding c2
    flat   = reshape @'[n, 4*4*50] p2
    f1     = relu $ linear fc1 flat
    logits = linear fc2 f1
    output = logSoftmax logits 1


main = undefined
