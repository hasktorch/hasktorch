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

fromNestedList :: [[Float]] -> Tensor
fromNestedList ls = asTensor ls

num2tensor :: Int -> Tensor
num2tensor = fromList . pad . (map fromIntegral) . toBinary

num2list :: Int -> [Float]
num2list = pad . (map fromIntegral) . toBinary


toBinary :: Int -> [Int]
toBinary 1 = [1]
toBinary 0 = [0]
toBinary n = (toBinary $ n `quot` 2) ++ [n `mod` 2]

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

oneHot :: Char -> Int -> Float
oneHot '+' lsNum = if lsNum == 10 then 1.0 else 0.0
oneHot n lsNum = if lsNum == (read [n]) then 1.0 else 0.0

toOneHot :: Char -> [Float]
toOneHot n = map (oneHot n) [0..10]

toOneHotSeq :: Int -> [[Float]]
toOneHotSeq n = map toOneHot (show n)


randomInt :: IO (Int)
randomInt = do
    n <- randn' []
    let n' = Prelude.abs $ toDouble n
    if (n' > 1.0)
        then return $ truncate (n' * 10)
        else return $ truncate (n' * 100)

randomSum :: IO ([Int])
randomSum = do
    a <- randomInt
    b <- randomInt
    return [a, b, a + b]

-- generate a n-length list of sequences
generate :: Int -> IO ([(Tensor, Tensor)])
generate n = mapM block [1..n]
  where
    block num = do
        sum <- randomSum
        let inputA = toOneHotSeq (sum !! 0)
        let op = toOneHot '+'
        let inputB = toOneHotSeq (sum !! 1)
        let output = fromNestedList $ toOneHotSeq (sum !! 2)
        let inputs = fromNestedList (inputA ++ [op] ++ inputB)
        return (inputs, output)
