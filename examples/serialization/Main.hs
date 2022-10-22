{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}

module Main where

import Codec.Serialise
import Control.Monad (foldM)
import GHC.Generics (Generic)
import Torch

batch_size = 64

num_iters = 2000

num_features = 3

instance Serialise Parameter where
  encode p = encode p' where p' :: [Float] = asValue $ toDependent p
  decode = IndependentTensor . (asTensor :: [Float] -> Tensor) <$> decode

deriving instance Serialise Linear

deriving instance Generic LinearSpec

deriving instance Serialise LinearSpec

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = 42.0 * ones' [num_features, 1]
    bias = 3.14 * ones' [1]

asFloats :: IndependentTensor -> [Float]
asFloats = (asValue :: Tensor -> [Float]) . toDependent

areSame :: Linear -> Linear -> Bool
areSame trained trained' =
  (asFloats (trained.bias) == asFloats (trained'.bias))
    && (asFloats (trained.weight) == asFloats (trained'.weight))

main :: IO ()
main = do
  init <- sample $ LinearSpec {in_features = num_features, out_features = 1}
  trained <- foldLoop init num_iters $ \state i -> do
    input <- randnIO' [batch_size, num_features]
    let expected_output = groundTruth input
        output = model state input
        loss = mseLoss expected_output output
        flat_parameters = flattenParameters state
        gradients = grad loss flat_parameters
    if i `mod` 100 == 0 then putStrLn $ "Loss: " ++ show loss else pure ()
    new_flat_parameters <-
      mapM makeIndependent $
        sgd 5e-3 flat_parameters gradients
    return $ replaceParameters state $ new_flat_parameters
  let bsl = serialise trained
      trained' :: Linear = deserialise bsl
  print $
    "Weights and biases survived a round trip through `serialize`: "
      <> show
        (areSame trained trained')
  pure ()
  where
    foldLoop x count block = foldM block x [1 .. count]
