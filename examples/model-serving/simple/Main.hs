{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import Network.Wai.Handler.Warp (run)
import Control.Monad.IO.Class (liftIO)
import Data.Aeson (FromJSON, ToJSON)
import GHC.Generics (Generic)
import Servant
import Torch 

data Result = Result {
  msg :: String, 
  result :: [Float]
  } deriving (Show, Generic) 

instance ToJSON Result
instance FromJSON Result

type HelloTorchAPI = "inference" :> Capture "value" Float :> Get '[JSON] [Result]

helloTorchH value = liftIO $ helloTorch value

helloTorch :: Float -> IO [Result]
helloTorch value = pure $ [Result "f(x) = 2.0 * x is " [result]]
  where
    t = asTensor value :: Tensor
    result = asValue (2.0 * t) :: Float

torchApi :: Proxy HelloTorchAPI
torchApi = Proxy

server :: Server HelloTorchAPI
server = helloTorchH

app :: Application
app = serve torchApi server

main :: IO ()
main = do
  putStrLn $ "Running server on port " ++ show port
  run port app
  where port = 8081
