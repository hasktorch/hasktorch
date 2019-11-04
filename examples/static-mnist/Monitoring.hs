{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}

module Monitoring where

import           GHC.TypeLits
import           Graphics.Vega.VegaLite

import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import           Torch.Typed.Tensor

type Iterate = Int
type Metrics device = [(Iterate, Metric device)]

data Metric (device :: (D.DeviceType, Nat))
 where
  Metric
    :: forall device
     . { trainingLoss :: Tensor device 'D.Float '[]
       , testLoss     :: Tensor device 'D.Float '[]
       , testError    :: Tensor device 'D.Float '[]
       }
    -> Metric device
 deriving Show

asFloat :: forall device . Tensor device 'D.Float '[] -> Float
asFloat t = D.asValue . toDynamic . toCPU $ t

plotLosses :: forall device . String -> Metrics device -> IO ()
plotLosses file metrics =
  toHtmlFile file $
  let enc = encoding
            . position X [ PName "Iterate", PmType Quantitative, axis ]
            . position Y [ PName "Loss",    PmType Quantitative ]
            . color [ MName "Lines", MmType Nominal ]
      enc2 = encoding
            . position X [ PName "Iterate",   PmType Quantitative, axis ]
            . position Y [ PName "ErrorRate", PmType Quantitative ]
            . color [ MName "Lines", MmType Nominal ]
      axis = PAxis [ AxValues (map (\(i,_) -> fromIntegral i) $ metrics)]
      dat = foldl
            (\sum' (iterate, Metric{..}) ->
               sum' .
               dataRow [ ("Iterate", Number (fromIntegral iterate))
                       , ("Lines",   Str "training loss")
                       , ("Loss",    Number (realToFrac (asFloat trainingLoss)))
                       ] .
               dataRow [ ("Iterate", Number (fromIntegral iterate))
                       , ("Lines",   Str "test loss")
                       , ("Loss",    Number (realToFrac (asFloat testLoss)))
                       ] .
               dataRow [ ("Iterate",   Number (fromIntegral iterate))
                       , ("Lines",     Str "test error rate")
                       , ("ErrorRate", Number (realToFrac (asFloat testError)))
                       ]
            )
            (dataFromRows [])
            metrics
  in toVegaLite [ dat []
                , hConcat [ asSpec [ mark Line []
                                   , enc []
                                   , height 300
                                   , width 400
                                   ]
                          , asSpec [ mark Line []
                                   , enc2 []
                                   , height 300
                                   , width 400
                                   ]
                          ]
                ] 

printLosses :: forall device . (Iterate, Metric device) -> IO ()
printLosses (i, Metric {..}) =
  let asFloat t = D.asValue . toDynamic . toCPU $ t :: Float
  in  putStrLn
        $  "Iteration: "
        <> show i
        <> ". Training batch loss: "
        <> show (asFloat trainingLoss)
        <> ". Test loss: "
        <> show (asFloat testLoss)
        <> ". Test error-rate: "
        <> show (asFloat testError)
