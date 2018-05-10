{-# LANGUAGE FlexibleContexts #-}
module Utils where

import Control.Monad.Trans
import Control.Monad.Trans.Maybe
import qualified Torch.Double.NN.Conv1d as NN1
import qualified Torch.Double.NN.Conv2d as NN2

import Torch.Double as Torch

-- Make a rank-1 tensor containing the cosine reshaped to the desired dimensionality.
mkCosineTensor :: forall d . Dimensions d => KnownNatDim (Product d) => MaybeT IO (Tensor d)
mkCosineTensor = do
  t <- MaybeT . pure . fromList $ [1..(fromIntegral $ natVal (Proxy :: Proxy (Product d)))]
  lift $ Torch.cos t

printFullConv1d :: KnownNat4 a b c d => String -> Conv1d a b c d -> IO ()
printFullConv1d title c = do
  putStrLn ""
  putStrLn "---------------------------------------"
  putStrLn title
  print c
  print (NN1.weights c)
  print (NN1.bias c)
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



