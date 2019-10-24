{-# LANGUAGE AllowAmbiguousTypes     #-}
{-# LANGUAGE ConstraintKinds         #-}
{-# LANGUAGE DataKinds               #-}
{-# LANGUAGE DeriveGeneric           #-}
{-# LANGUAGE FlexibleContexts        #-}
{-# LANGUAGE FlexibleInstances       #-}
{-# LANGUAGE GADTs                   #-}
{-# LANGUAGE MultiParamTypeClasses   #-}
{-# LANGUAGE NoStarIsType            #-}
{-# LANGUAGE OverloadedLists         #-}
{-# LANGUAGE PolyKinds               #-}
{-# LANGUAGE RankNTypes              #-}
{-# LANGUAGE RecordWildCards         #-}
{-# LANGUAGE ScopedTypeVariables     #-}
{-# LANGUAGE TypeApplications        #-}
{-# LANGUAGE TypeFamilies            #-}
{-# LANGUAGE TypeOperators           #-}
{-# LANGUAGE UndecidableInstances    #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module LSTM where

import           Control.Monad                  ( foldM
                                                , when
                                                , void
                                                )
import           Control.Exception.Safe         ( try
                                                , SomeException(..)
                                                )
import           Data.List                      ( foldl'
                                                , intersperse
                                                , scanl'
                                                )
import           Data.Reflection
import           Foreign.ForeignPtr
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           System.Environment
import           System.IO.Unsafe
import           Prelude                 hiding ( tanh )
import qualified ATen.Cast                     as ATen
import qualified ATen.Class                    as ATen
import qualified ATen.Type                     as ATen
import qualified ATen.Managed.Type.Tensor      as ATen
import qualified Torch.Autograd                as A
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.NN                      as A
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import           Torch.Typed
import           Torch.Typed.Factories
import           Torch.Typed.Native      hiding ( linear )
import           Torch.Typed.NN
import           Torch.Typed.NN.Recurrent.Cell.LSTM

run
    :: forall d i h b
     . (KnownDType d, KnownNat i, KnownNat h, KnownNat b)
    => [(Tensor d '[b, i])]
    -> Tensor d '[b, h]
    -> LSTMCell d i h
    -> Int
    -> IO (LSTMCell d i h)
run input_tensors expected_output model i = do

    let output = snd $ forward
            model
            ((zeros, zeros) :: (Tensor d '[b, h], Tensor d '[b, h]))
            input_tensors
    let loss = mse_loss output expected_output

    print loss

    let flat_parameters = A.flattenParameters model
    let gradients       = A.grad (toDynamic loss) flat_parameters
    -- new parameters returned by the SGD update functions
    new_flat_parameters <- mapM A.makeIndependent
        $ A.sgd 5e-2 flat_parameters gradients

    -- return the new model state "to" the next iteration of foldLoop
    return $ A.replaceParameters model new_flat_parameters

toBackend
    :: forall t . (ATen.Castable t (ForeignPtr ATen.Tensor)) => String -> t -> t
toBackend backend t = unsafePerformIO $ case backend of
    "CUDA" -> ATen.cast1 ATen.tensor_cuda t
    _      -> ATen.cast1 ATen.tensor_cpu t

main :: IO ()
main = do
    backend' <- try (getEnv "BACKEND") :: IO (Either SomeException String)
    let backend = case backend' of
            Right "CUDA" -> "CUDA"
            _            -> "CPU"
    let foldLoop x count block = foldM @[] block x [1 .. count]

    -- Memorization test:  just check to see that the cell learns to recall the last value of the expected sequence
    (input_tensor :: [Tensor D.Float '[1, 2]]) <- mapM
        (const (randn :: IO (Tensor D.Float '[1, 2])))
        [0 .. 4]
    expected_output                 <- randn :: IO (Tensor D.Float '[1, 2])

    (init :: LSTMCell 'D.Float 2 2) <- A.sample LSTMCellSpec
    init'                           <- A.replaceParameters init <$> traverse
        (A.makeIndependent . toBackend backend . A.toDependent)
        (A.flattenParameters init)
    putStrLn "\nLSTM Training Loop"
    -- training loop 
    foldLoop init' 10 (run input_tensor expected_output)
    return ()
