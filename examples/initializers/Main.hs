module Main where

import Torch.Tensor
import Torch.TensorFactories

data NonLinearity = Linear | Sigmoid | Tanh | Relu | LeakyRelu

data Fan = FanIn | FanOut

calculateGain :: NonLinearity -> Maybe Float -> Float
calculateGain Linear _ = 1.0
calculateGain Sigmoid _ = 1.0
calculateGain Tanh _ = 5.0 / 3
calculateGain LeakyRelu param = sqrt (2.0 / (1.0 + (negativeSlope param) ^^ 2))
    where
        negativeSlope Nothing = 0.01
        negativeSlope (Just value) = value

calculateFan :: Tensor -> Fan -> Float
calculateFan t mode = undefined

main :: IO ()
main = do
    putStrLn "Linear default gain"
    print $ calculateGain LeakyRelu (Just $ sqrt 5.0)
    putStrLn "Done"
