{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Typed.NN.Recurrent.GRUSpec
  ( Torch.Typed.NN.Recurrent.GRUSpec.spec,
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
import Torch.Typed.NN.Recurrent.GRU
import Torch.Typed.Parameter

spec :: Spec
spec = return ()

testGRU ::
  IO
    ( HList
        '[ Parameter '( 'D.CPU, 0) 'D.Float '[21, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21]
         ]
    )
testGRU = do
  let spec = GRUSpec @5 @7 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (DropoutSpec 0.1)
  model <- A.sample spec
  pure . flattenParameters $ model

testGRUWithConstInitSpec ::
  IO
    ( HList
        '[ Parameter '( 'D.CPU, 0) 'D.Float '[21, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21]
         ]
    )
testGRUWithConstInitSpec = do
  let spec = GRUSpec @5 @7 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (DropoutSpec 0.1)
      spec' = GRUWithConstInitSpec spec Torch.Typed.Factories.zeros
  model <- A.sample spec'
  pure . flattenParameters $ model

testGRUWithLearnedInitSpec ::
  IO
    ( HList
        '[ Parameter '( 'D.CPU, 0) 'D.Float '[21, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 5],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 14],
           Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[21],
           Parameter '( 'D.CPU, 0) 'D.Float '[6, 7]
         ]
    )
testGRUWithLearnedInitSpec = do
  let spec = GRUSpec @5 @7 @3 @'Bidirectional @'D.Float @'( 'D.CPU, 0) (DropoutSpec 0.1)
      spec' = GRUWithLearnedInitSpec spec Torch.Typed.Factories.zeros
  model <- A.sample spec'
  pure . flattenParameters $ model
