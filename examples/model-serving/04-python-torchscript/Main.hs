

{-# LANGUAGE OverloadedStrings #-}
import Torch
import Torch.Script as S

main  = do
  -- load parameters
  S.IVGenericDict params <- pickleLoad "mnist.dict.pt"

  -- load example data
  S.IVGenericDict example <- pickleLoad "mnist.example.pt"
  let ivt = snd $ example !! 0
  let S.IVTensor t = ivt

  -- load torchscript
  tsModule <- S.load WithoutRequiredGrad "mnist.ts.pt"
  
  let result = forward tsModule [ivt] 
  print result

  putStrLn "Done"
