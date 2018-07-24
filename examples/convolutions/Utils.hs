{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Utils where

import Numeric.Dimensions
import Data.Singletons.Prelude.List hiding (All)
import Data.Singletons.TypeLits

import qualified Torch.Double.NN.Conv1d as NN1
import qualified Torch.Double.NN.Conv2d as NN2

import Torch.Double as Torch

-- Make a rank-1 tensor containing the cosine reshaped to the desired dimensionality.
mkCosineTensor
  :: forall d
  . Dimensions d
  => KnownDim (Product d)
  => KnownNat (Product d)
  => Maybe (Tensor d)
mkCosineTensor =
  Torch.cos <$> fromList [1..fromIntegral $ dimVal (dim :: Dim (Product d))]

printFullConv1d
  :: All KnownNat '[a,c,b,d]
  => All KnownDim '[a,b,c,d,a*c]
  => String
  -> Conv1d a b c d
  -> IO ()
printFullConv1d = printConvs NN1.weights NN1.bias

printFullConv2d
  :: All KnownDim '[a,b,c,d,a*c]
  => String
  -> Conv2d a b '(c,d)
  -> IO ()
printFullConv2d = printConvs NN2.weights NN2.bias

printConvs :: (Show layer, Show weight, Show bias) => (layer -> weight) -> (layer -> bias) -> String -> layer -> IO ()
printConvs toWeights toBias title layer = do
  putStrLn ""
  putStrLn "---------------------------------------"
  putStrLn title
  print layer
  print (toWeights layer)
  print (toBias layer)
  putStrLn "---------------------------------------"

section :: String -> IO () -> IO ()
section header rest = do
  putStrLn ""
  putStrLn ""
  putStrLn "======================================="
  putStrLn ("SECTION:  " ++ header)
  putStrLn "======================================="
  rest
  putStrLn ("END OF SECTION: " ++ header)
  putStrLn ""



