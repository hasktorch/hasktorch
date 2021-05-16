{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Network.Wai.Handler.Warp (run)
import Control.Monad.IO.Class (liftIO)
import Data.Aeson (FromJSON, ToJSON)
import GHC.Generics (Generic)
import Servant
import Torch 
import Model (model, train)
import Serialise

import Codec.Serialise (serialise, deserialise)

data Result = Result {
  msg :: String, 
  result :: [Float]
  } deriving (Show, Generic) 

instance ToJSON Result
instance FromJSON Result

type InferAPI = "inference" 
  :> Capture "arg1" Float 
  :> Capture "arg2" Float 
  :> Capture "arg3" Float 
  :> Get '[JSON] [Result]

torchApi :: Proxy InferAPI
torchApi = Proxy

wrapModel trained x1 x2 x3 = liftIO $ 
  pure $ [Result "infer" [asValue (model trained (asTensor [x1, x2, x3])) :: Float]]

main :: IO ()
main = do
  putStrLn $ "Training model"
  trained <- train
  putStrLn $ "Test " ++ show (asValue $ model trained (asTensor $ [1.0, 2.0, 3.0 :: Float]) :: Float)

  let bsl = serialise trained
  print bsl
  let trained' :: Linear = deserialise bsl

  print trained
  print trained'
  putStrLn $ "Running server on port " ++ show port
  -- run port (serve torchApi (wrapModel trained'))
  where port = 8081
