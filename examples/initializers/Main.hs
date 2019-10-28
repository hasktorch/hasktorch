module Main where

import Torch.Functions hiding (sqrt)
import Torch.Tensor
import Torch.TensorFactories

data NonLinearity = Linear | Sigmoid | Tanh | Relu | LeakyRelu

data FanMode = FanIn | FanOut

newtype Shape = Shape [Int]

-- | Gain scaling value for He initialization
calculateGain :: NonLinearity -> Maybe Float -> Float
calculateGain Linear _ = 1.0
calculateGain Sigmoid _ = 1.0
calculateGain Tanh _ = 5.0 / 3
calculateGain Relu _ = sqrt 2.0
calculateGain LeakyRelu param = sqrt (2.0 / (1.0 + (negativeSlope param) ^^ 2))
    where
        negativeSlope Nothing = 0.01
        negativeSlope (Just value) = value

-- | Fan-in / Fan-out scaling calculation for He Initialization
calculateFan :: [Int] -> (Int, Int)
calculateFan shape =
    if dimT < 2 then
        error "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
    else if dimT == 2 then
        (shape !! 1, shape !! 0)
        else 
            (numInputFmaps * receptiveFieldSize,
            numOutputFmaps * receptiveFieldSize)
    where
        dimT = length shape
        numInputFmaps = shape !! 1 -- size t 1
        numOutputFmaps = shape !! 0 -- size t 0
        receptiveFieldSize = product $ tail shape

getter :: FanMode -> ((Int, Int) -> Int)
getter FanIn = fst
getter FanOut = snd

-- | Xavier Initialization - Uniform
xavierUniform :: Float -> [Int] -> IO Tensor
xavierUniform gain shape = do
    init <- rand' shape
    pure $ subScalar (mulScalar init (bound * 2.0)) bound
    where
        (fanIn, fanOut) = calculateFan shape
        std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
        bound = sqrt 3.0 * std

-- | Xavier Initialization - Normal
xavierNormal :: Float -> [Int] -> IO Tensor
xavierNormal gain shape = do
    init <- randn' shape
    pure $ mulScalar init std
    where
        (fanIn, fanOut) = calculateFan shape
        std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))

-- | Kaiming Initialization - Uniform
kaimingUniform :: Float -> FanMode -> NonLinearity -> [Int] -> IO Tensor
kaimingUniform a mode nonlinearity shape = do
    init <- rand' shape
    pure $ subScalar (mulScalar init (bound * 2.0)) bound
    where 
        gain = calculateGain nonlinearity (Just a)
        fanValue = fromIntegral $ (getter mode) (calculateFan shape)
        std = gain / (sqrt fanValue)
        bound = (sqrt 3.0) * std

-- | Kaiming Initialization - Normal
kaimingNormal :: Float -> FanMode -> NonLinearity -> [Int] -> IO Tensor
kaimingNormal a mode nonlinearity shape = do
    init <- (randn' shape)
    pure $ mulScalar init std
    where 
        gain = calculateGain nonlinearity (Just a)
        fanValue = fromIntegral $ (getter mode) (calculateFan shape)
        std = gain / (sqrt fanValue)

-- PyTorch defaults
kaimingUniform' :: [Int] -> IO Tensor
kaimingUniform' = kaimingUniform 0.0 FanIn LeakyRelu

kaimingNormal' :: [Int] -> IO Tensor
kaimingNormal' = kaimingNormal 0.0 FanIn LeakyRelu

xavierUniform' :: [Int] -> IO Tensor
xavierUniform' = xavierUniform 1.0

xavierNormal' :: [Int] -> IO Tensor
xavierNormal' = xavierNormal 1.0

main :: IO ()
main = do
    putStrLn "\nKaiming Uniform"
    x <- kaimingUniform' [4, 5]
    print x
    putStrLn "\nKaiming Normal"
    x <- kaimingNormal' [4, 5]
    print x
    putStrLn "\nXavier Uniform"
    x <- xavierUniform' [4, 5]
    print x
    putStrLn "\nXavier Normal"
    x <- xavierNormal' [4, 5]
    print x
    putStrLn "Done"
