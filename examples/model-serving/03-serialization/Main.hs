{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import Control.Monad.IO.Class (liftIO)
import Data.Aeson (FromJSON, ToJSON)
import GHC.Generics (Generic)
import Model (model, train)
import Network.Wai.Handler.Warp (run)
import Servant
import Torch
import Torch.Serialize

data Result = Result
  { msg :: String,
    result :: [Float]
  }
  deriving (Show, Generic)

instance ToJSON Result

instance FromJSON Result

type InferAPI =
  "inference"
    :> Capture "arg1" Float
    :> Capture "arg2" Float
    :> Capture "arg3" Float
    :> Get '[JSON] [Result]

torchApi :: Proxy InferAPI
torchApi = Proxy

wrapModel trained x1 x2 x3 =
  liftIO $
    pure $ [Result "infer" [asValue (model trained (asTensor [x1, x2, x3])) :: Float]]

main :: IO ()
main = do
  putStrLn $ "Training model"
  init <- sample $ LinearSpec {in_features = 3, out_features = 1}
  trained <- train init
  putStrLn $ "Test " ++ show (asValue $ model trained (asTensor $ [1.0, 2.0, 3.0 :: Float]) :: Float)

  saveParams trained "trained.pt"
  trained' <- loadParams init "trained.pt"

  putStrLn "\nSaved  model parameters:"
  print trained
  putStrLn "\nLoaded model parameters:"
  print trained'
  putStrLn $ "\nRunning server on port " ++ show port
  putStrLn $ "\nTry a model inference:\n\nhttp://localhost:" ++ show port ++ "/inference/1.0/2.0/3.0/"

  run port (serve torchApi (wrapModel trained'))
  where
    port = 8081
