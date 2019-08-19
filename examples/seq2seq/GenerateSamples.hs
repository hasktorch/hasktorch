{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}

module GenerateSamples where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN
import GHC.Generics

-- | convert a list to a one-dimensional tensor
fromList :: [Float] -> Tensor
fromList ls = asTensor ls


num2tensor :: Int -> Tensor
num2tensor = fromList . pad . (map fromIntegral) . toBinary

-- tensor2num :: Tensor -> Int
-- tensor2num t =  

pad :: [Float] -> [Float]
pad ls = case (length ls) of
    1 -> [0, 0, 0, 0, 0, 0, 0] ++ ls
    2 -> [0, 0, 0, 0, 0, 0] ++ ls
    3 -> [0, 0, 0, 0, 0] ++ ls 
    4 -> [0, 0, 0, 0] ++ ls
    5 -> [0, 0, 0] ++ ls
    6 -> [0, 0] ++ ls
    7 -> [0] ++ ls
    otherwise -> ls


toBinary :: Int -> [Int] 
toBinary 1 = [1]
toBinary 0 = [0]
toBinary n = (toBinary $ n `quot` 2) ++ [n `mod` 2]


randomInt :: IO (Int)
randomInt = do
    n <- randn' []
    let n' = Prelude.abs $ toDouble n
    if (n' > 1.0) 
        then return $ truncate (n' * 10)
        else return $ truncate (n' * 100)
