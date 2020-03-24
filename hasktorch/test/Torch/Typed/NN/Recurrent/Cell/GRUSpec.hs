{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DataKinds #-}

module Torch.Typed.NN.Recurrent.Cell.GRUSpec
  ( Torch.Typed.NN.Recurrent.Cell.GRUSpec.spec
  )
where

import           Test.Hspec

import           Torch.HList
import qualified Torch.NN                      as A
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import           Torch.Typed.Factories
import           Torch.Typed.Parameter
import           Torch.Typed.NN.Recurrent.Cell.GRU

spec :: Spec
spec = return ()

testGRUCell
  :: IO
       (HList
          '[Parameter '( 'D.CPU, 0) 'D.Float '[21, 5],
            Parameter '( 'D.CPU, 0) 'D.Float '[21, 7],
            Parameter '( 'D.CPU, 0) 'D.Float '[21],
            Parameter '( 'D.CPU, 0) 'D.Float '[21]])

testGRUCell = do
  let spec = GRUCellSpec  @5 @7 @'D.Float @'( 'D.CPU, 0)
  model <- A.sample spec
  pure . flattenParameters $ model