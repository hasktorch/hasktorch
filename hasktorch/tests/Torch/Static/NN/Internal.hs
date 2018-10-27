{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Torch.Static.NN.Internal where

import Numeric.Dimensions
import Data.Singletons.Prelude.List hiding (All)
import Data.Singletons.TypeLits

import Control.Monad.Trans
import Control.Monad.Trans.Maybe
import qualified Torch.Double.NN.Conv1d as NN1
import qualified Torch.Double.NN.Conv2d as NN2

import Torch.Double as Torch

-- Make a rank-1 tensor containing the cosine reshaped to the desired dimensionality.
mkCosineTensor
  :: forall d
  . Dimensions d
  => KnownDim (Product d)
  => KnownNat (Product d)
  => MaybeT IO (Tensor d)
mkCosineTensor = do
  t <- MaybeT . pure . fromList $ [1..fromIntegral $ dimVal (dim :: Dim (Product d))]
  lift $ Torch.cos t

printFullConv1d
  :: All KnownNat '[a,c,b,d]
  => All KnownDim '[a,b,c,d,a*c]
  => String -> Conv1d a b c d -> IO ()
printFullConv1d title c = do
  putStrLn ""
  putStrLn "---------------------------------------"
  putStrLn title
  print c
  print (NN1.weights c)
  print (NN1.bias c)
  putStrLn "---------------------------------------"

printFullConv2d :: All KnownDim '[a,b,c,d,a*c] => String -> Conv2d a b c d -> IO ()
printFullConv2d title c = do
  putStrLn ""
  putStrLn "---------------------------------------"
  putStrLn title
  print c
  print (NN2.weights c)
  print (NN2.bias c)
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



