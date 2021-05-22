

{-# LANGUAGE OverloadedStrings #-}
import Torch
import Torch.Script as S

loadFile file = do
  print file
  S.IVGenericDict vals <- pickleLoad file
  print vals
  pure vals

modelFile = "test2sd.pt"

main = do
  vals <- loadFile modelFile
  let (_, IVTensor weight) = vals !! 0
  let (_, IVTensor bias) = vals !! 1
  weight' <- makeIndependent weight
  bias' <- makeIndependent bias
  -- init <- sample $ LinearSpec { in_features = 2, out_features = 1 }
  let model = Linear weight' bias'
  let x = asTensor [1.0 :: Float, 2.0]
  print model
  print $ linear model x
  putStrLn "Done"
