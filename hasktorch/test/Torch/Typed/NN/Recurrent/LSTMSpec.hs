{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Typed.NN.Recurrent.LSTMSpec
  ( Torch.Typed.NN.Recurrent.LSTMSpec.spec,
  )
where

import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.NN as A
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.NN
import Torch.Typed.NN.Recurrent.Auxiliary
import Torch.Typed.NN.Recurrent.LSTM
import Torch.Typed.Parameter

spec :: Spec
spec = return ()

testLSTM ::
  IO
    ( HList
        '[ Parameter '( 'D.CPU, 0) 'D.Float '[28, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28]
         ]
    )
testLSTM = do
  let spec = LSTMSpec @5 @7 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (DropoutSpec 0.1)
  model <- A.sample spec
  pure . flattenParameters $ model

testLSTMWithConstInitSpec ::
  IO
    ( HList
        '[ Parameter '( 'D.CPU, 0) 'D.Float '[28, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28]
         ]
    )
testLSTMWithConstInitSpec = do
  let spec = LSTMSpec @5 @7 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (DropoutSpec 0.1)
      spec' = LSTMWithConstInitSpec spec Torch.Typed.Factories.zeros Torch.Typed.Factories.zeros
  model <- A.sample spec'
  pure . flattenParameters $ model

testLSTMWithLearnedInitSpec ::
  IO
    ( HList
        '[ Parameter '( 'D.CPU, 0) 'D.Float '[28, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[6, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[6, 7]
         ]
    )
testLSTMWithLearnedInitSpec = do
  let spec = LSTMSpec @5 @7 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (DropoutSpec 0.1)
      spec' = LSTMWithLearnedInitSpec spec Torch.Typed.Factories.zeros Torch.Typed.Factories.zeros
  model <- A.sample spec'
  pure . flattenParameters $ model
