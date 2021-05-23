

{-# LANGUAGE OverloadedStrings #-}
import Torch
import Torch.Script as S

testSimple = do
  S.IVGenericDict vals <- pickleLoad "test2sd.pt"
  let (_, IVTensor weight) = vals !! 0
  let (_, IVTensor bias) = vals !! 1
  weight' <- makeIndependent weight
  bias' <- makeIndependent bias
  let model = Linear weight' bias'
  let x = asTensor [1.0 :: Float, 2.0]
  print model
  print $ linear model x

main  = do
  -- load parameters
  S.IVGenericDict params <- pickleLoad "mnist.pt"

  -- load example data
  S.IVGenericDict example <- pickleLoad "mnist.example.pt"
  let ivt = snd $ example !! 0
  let S.IVTensor t = ivt

  -- load torchscript
  tsModule <- S.load WithoutRequiredGrad "mnist.ts.pt"
  
  -- compute forward function with torchscript
  let result = S.forward tsModule [ivt] 
  print result

  putStrLn "Done"
