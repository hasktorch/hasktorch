{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

import Torch.Tensor
import Torch.TensorFactories (eye', ones', rand', randn', zeros')
import Torch.Functions

rosenbrock :: Float -> Float -> Tensor -> Tensor -> Tensor
rosenbrock a b x y = square (cadd (cmul x (-1.0 :: Float)) a) + cmul (square (y - square x)) b
    where square c = pow c (2 :: Int)

rosenbrock' = rosenbrock 1.0 100.0

main = do
    print $ rosenbrock' (asTensor ([-2.0, -1.0, 0.0, 1.0, 2.0] :: [Float])) 0.0
    print $ rosenbrock' (asTensor ([-2.0, -1.0, 0.0, 1.0, 2.0] :: [Float])) 1.0
    print $ rosenbrock' (asTensor ([-2.0, -1.0, 0.0, 1.0, 2.0] :: [Float])) 2.0
    putStrLn "Done"

