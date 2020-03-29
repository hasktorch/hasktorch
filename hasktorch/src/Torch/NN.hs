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
import System.IO.Unsafe (unsafePerformIO)

import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen

import Torch.Internal.Cast (cast3, cast6, cast14)

import Torch.Autograd
import Torch.Initializers
import Torch.Tensor
import Torch.TensorFactories (ones', randIO', randnIO')
import Torch.Functional
import GHC.Generics

type Parameter = IndependentTensor
type ParamStream a = State [Parameter] a

nextParameter :: ParamStream Parameter
nextParameter = do
  params <- get
  case params of
    [] -> error "Not enough parameters supplied to replaceParameters"
    (p : t) -> do put t; return p

class Parameterized f where
  flattenParameters :: f -> [Parameter]
  default flattenParameters :: (Generic f, Parameterized' (Rep f)) => f -> [Parameter]
  flattenParameters f = flattenParameters' (from f)

  replaceOwnParameters :: f -> ParamStream f
  default replaceOwnParameters :: (Generic f, Parameterized' (Rep f)) => f -> ParamStream f
  replaceOwnParameters f = to <$> replaceOwnParameters' (from f)

instance Parameterized Parameter where
  flattenParameters = pure
  replaceOwnParameters _ = nextParameter

instance Parameterized Double where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized [Int] where
  flattenParameters _ = []
  replaceOwnParameters = return

instance Parameterized (Tensor -> Tensor) where
  flattenParameters _ = []
  replaceOwnParameters = return

class Parameterized' f where
  flattenParameters' :: f a -> [Parameter]
  replaceOwnParameters' :: f a -> ParamStream (f a)

instance Parameterized' U1 where
  flattenParameters' U1 = []
  replaceOwnParameters' U1 = return U1

instance (Parameterized' f, Parameterized' g) => Parameterized' (f :+: g) where
  flattenParameters' (L1 x) = flattenParameters' x
  flattenParameters' (R1 x) = flattenParameters' x
  replaceOwnParameters' (L1 x) = do
    x' <- replaceOwnParameters' x
    return $ L1 x'
  replaceOwnParameters' (R1 x) = do
    x' <- replaceOwnParameters' x
    return $ R1 x'

instance (Parameterized' f, Parameterized' g) => Parameterized' (f :*: g) where
  flattenParameters' (x :*: y) = flattenParameters' x ++ flattenParameters' y
  replaceOwnParameters' (x :*: y) = do
    x' <- replaceOwnParameters' x
    y' <- replaceOwnParameters' y
    return $ x' :*: y'

instance (Parameterized c) => Parameterized' (K1 i c) where
  flattenParameters' (K1 x) = flattenParameters x
  replaceOwnParameters' (K1 x) = do
    x' <- replaceOwnParameters x
    return $ K1 x'

instance (Parameterized' f) => Parameterized' (M1 i t f) where
  flattenParameters' (M1 x) = flattenParameters' x
  replaceOwnParameters' (M1 x) = do
    x' <- replaceOwnParameters' x
    return $ M1 x'

replaceParameters :: Parameterized f => f -> [Parameter] -> f
replaceParameters f params =
  let (f', remaining) = runState (replaceOwnParameters f) params in
  if null remaining
    then f'
    else error "Some parameters in a call to replaceParameters haven't been consumed!"

class Randomizable spec f | spec -> f where
  sample :: spec -> IO f

class (Randomizable spec f, Parameterized f) => Module spec f

data LinearSpec = LinearSpec { in_features :: Int, out_features :: Int }
  deriving (Show, Eq)

data Linear = Linear { weight :: Parameter, bias :: Parameter } deriving (Show, Generic)

linear :: Linear -> Tensor -> Tensor
linear layer input = linear' input w b
    where
        linear' input weight bias = unsafePerformIO $ (cast3 ATen.linear_ttt) input weight bias
        w = toDependent (weight layer)
        b = toDependent (bias layer)

instance Randomizable LinearSpec Linear where
  sample LinearSpec{..} = do
      w <- makeIndependent =<< kaimingUniform' [out_features, in_features]
      -- w <- makeIndependent =<< randn' [out_features, in_features]
      b <- makeIndependent =<< randnIO' [out_features]
      return $ Linear w b

instance Parameterized Linear
-- This instance generates following codes.
--
---------------------------------------------------
-- instance Parameterized Linear where
--   flattenParameters Linear{..} = [weight, bias]
--   replaceOwnParameters _ = do
--     weight <- nextParameter
--     bias <- nextParameter
--     return $ Linear{..}

instance Parameterized [Linear]

data RNNParams = RNNParams {
	weightIH :: Tensor,
	weightHH :: Tensor,
	biasIH :: Tensor,
	biasHH :: Tensor
} deriving (Show)

-- | A long short-term memory (LSTM) cell.
stmCell 
    :: Tensor -- ^ input-hidden weights (4*hidden_size, input_size)
    -> Tensor -- ^ hidden-hidden weights (4*hidden_size, hidden_size)
    -> Tensor -- ^ input-hidden bias (4*hidden_size)
    -> Tensor -- ^ hidden-hidden bias, of shape (4*hidden_size)
    -> [Tensor] -- ^ hidden state
    -> Tensor -- ^ input
    -> (Tensor, Tensor) -- next hidden state, next cell state
lstmCell _w_ih _w_hh _b_ih _b_hh _hx _input =
    unsafePerformIO $
        (cast6 ATen.lstm_cell_tltttt) 
        _input _hx _w_ih _w_hh _b_ih _b_hh

-- | A gated recurrent unit (GRU) cell
gruCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
gruCell _w_ih _w_hh _b_ih _b_hh _hx _input =
  unsafePerformIO $
    (cast6 ATen.gru_cell_tttttt) 
    _input _hx _w_ih _w_hh _b_ih _b_hh

-- | An Elman RNN cell with tanh non-linearity
rnnTanhCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
rnnTanhCell _w_ih _w_hh _b_ih _b_hh _hx _input =
  unsafePerformIO $ (cast6 ATen.rnn_tanh_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

-- | An Elman RNN cell with ReLU non-linearity
rnnReluCell 
    :: Tensor -- ^ input-hidden weights
    -> Tensor -- ^ hidden-hidden weights
    -> Tensor -- ^ input-hidden bias
    -> Tensor -- ^ hidden-hidden bias
    -> Tensor -- ^ hidden state
    -> Tensor -- ^ input
    -> Tensor -- ^ output
rnnReluCell _w_ih _w_hh _b_ih _b_hh _hx _input =
  unsafePerformIO $ (cast6 ATen.rnn_relu_cell_tttttt) _input _hx _w_ih _w_hh _b_ih _b_hh

quantizedLstmCell 
	:: Tensor 
	-> Tensor 
	-> Tensor 
	-> Tensor 
	-> Tensor 
	-> Tensor 
	-> Tensor 
	-> Tensor 
	-> Float 
	-> Float 
	-> Float 
	-> Float 
	-> [Tensor] 
	-> Tensor 
	-> (Tensor,Tensor)
quantizedLstmCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh _hx _input =
  unsafePerformIO $ (cast14 ATen.quantized_lstm_cell_tlttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

quantizedGruCell 
    :: Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Float 
    -> Float 
    -> Float 
    -> Float 
    -> Tensor 
    -> Tensor 
    -> Tensor
quantizedGruCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh _hx _input =
  unsafePerformIO $ (cast14 ATen.quantized_gru_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

quantizedRnnReluCell 
    :: Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Float 
    -> Float 
    -> Float 
    -> Float 
    -> Tensor 
    -> Tensor 
    -> Tensor
quantizedRnnReluCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh _hx _input =
  unsafePerformIO $ (cast14 ATen.quantized_rnn_relu_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh

quantizedRnnTanhCell
    :: Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Tensor 
    -> Float 
    -> Float 
    -> Float 
    -> Float 
    -> Tensor 
    -> Tensor 
    -> Tensor
quantizedRnnTanhCell _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh _hx _input =
  unsafePerformIO $ (cast14 ATen.quantized_rnn_tanh_cell_ttttttttttssss) _input _hx _w_ih _w_hh _b_ih _b_hh _packed_ih _packed_hh _col_offsets_ih _col_offsets_hh _scale_ih _scale_hh _zero_point_ih _zero_point_hh
