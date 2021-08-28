{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.GRU where

import GHC.Generics
import Torch

data GRUSpec = GRUSpec
  { inputSize :: Int,
    hiddenSize :: Int
  }
  deriving (Eq, Show)

data GRUCell = GRUCell
  { weightsIH :: Parameter,
    weightsHH :: Parameter,
    biasIH :: Parameter,
    biasHH :: Parameter
  }
  deriving (Generic, Show)

gruCellForward ::
  -- | cell parameters
  GRUCell ->
  -- | input
  Tensor ->
  -- | hidden
  Tensor ->
  -- | output
  Tensor
gruCellForward GRUCell {..} input hidden =
  gruCell weightsIH' weightsHH' biasIH' biasHH' hidden input
  where
    weightsIH' = toDependent weightsIH
    weightsHH' = toDependent weightsHH
    biasIH' = toDependent biasIH
    biasHH' = toDependent biasHH

instance Parameterized GRUCell

instance Randomizable GRUSpec GRUCell where
  sample GRUSpec {..} = do
    -- https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
    weightsIH' <- makeIndependent =<< initScale <$> randIO' [3 * hiddenSize, inputSize]
    weightsHH' <- makeIndependent =<< initScale <$> randIO' [3 * hiddenSize, hiddenSize]
    biasIH' <- makeIndependent =<< initScale <$> randIO' [3 * hiddenSize]
    biasHH' <- makeIndependent =<< initScale <$> randIO' [3 * hiddenSize]
    pure $
      GRUCell
        { weightsIH = weightsIH',
          weightsHH = weightsHH',
          biasIH = biasIH',
          biasHH = biasHH'
        }
    where
      scale = Prelude.sqrt $ 1.0 / fromIntegral hiddenSize :: Float
      initScale = subScalar scale . mulScalar scale . mulScalar (2.0 :: Float)
