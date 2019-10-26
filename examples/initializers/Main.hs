module Main where

import Torch.Tensor
import Torch.TensorFactories

data NonLinearity = Linear | Sigmoid | Tanh | Relu | LeakyRelu

data FanMode = FanIn | FanOut

-- | Gain scaling value for He initialization
calculateGain :: NonLinearity -> Maybe Float -> Float
calculateGain Linear _ = 1.0
calculateGain Sigmoid _ = 1.0
calculateGain Tanh _ = 5.0 / 3
calculateGain LeakyRelu param = sqrt (2.0 / (1.0 + (negativeSlope param) ^^ 2))
    where
        negativeSlope Nothing = 0.01
        negativeSlope (Just value) = value

-- | Fan-in / Fan-out scaling calculation for He Initialization
calculateFan :: Tensor -> FanMode -> (Int, Int)
calculateFan t mode =
    if dim t < 2 then
        error "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
    else if dim t == 2 then
        (size t 1, size t 0)
        else 
            (numInputFmaps * receptiveFieldSize,
            numOutputFmaps * receptiveFieldSize)
    where
        numInputFmaps = size t 1
        numOutputFmaps = size t 0
        receptiveFieldSize = numel (select t 0 0)

-- | He initialization (TODO - needs checking)
kaimingUniform :: Tensor -> Float -> FanMode -> NonLinearity -> Tensor
kaimingUniform t a mode nonlinearity = undefined
    where 
        gain = calculateGain nonlinearity (Just a)
        getter FanIn = fst
        getter FanOut = snd
        fanValue = fromIntegral $ (getter mode) (calculateFan t mode)
        std = gain / (sqrt fanValue)
        bound = (sqrt 3.0) * std

main :: IO ()
main = do
    putStrLn "Linear default gain"
    print $ calculateGain LeakyRelu (Just $ sqrt 5.0)
    let foo = asTensor [[1.0 :: Float, 20.0]]
    print $ calculateFan foo FanIn
    let bar = asTensor [[[1.0 :: Float, 20.0]]]
    print $ calculateFan bar FanIn
    putStrLn "Done"
