module Main where

data NonLinearity = Linear | Sigmoid | Tanh | Relu | LeakyRelu

calculateGain Linear _ = 1.0
calculateGain Sigmoid _ = 1.0
calculateGain Tanh _ = 5.0 / 3
calculateGain LeakyRelu param = sqrt (2.0 / (1.0 + (negativeSlope param) ^^ 2))
    where
        negativeSlope Nothing = 0.01
        negativeSlope (Just value) = value

main = do
    print "Done"
