{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Typed.NN.Recurrent.Cell.LSTMSpec
  ( Torch.Typed.NN.Recurrent.Cell.LSTMSpec.spec,
  )
where

import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import qualified Torch.NN as A
import Torch.Typed.Factories
import Torch.Typed.NN.Recurrent.Cell.LSTM
import Torch.Typed.Parameter

spec :: Spec
spec = return ()

testLSTMCell ::
  IO
    ( HList
        '[ Parameter '( 'D.CPU, 0) 'D.Float '[28, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[28, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[28],
           Parameter '( 'D.CPU, 0) 'D.Float '[28]
         ]
    )
testLSTMCell = do
  let spec = LSTMCellSpec @5 @7 @'D.Float @'( 'D.CPU, 0)
  model <- A.sample spec
  pure . flattenParameters $ model
