{-# LANGUAGE OverloadedStrings #-}

import Torch
import Torch.Script as S

main = do
  -- load parameters
  S.IVGenericDict params <- pickleLoad "mnist.dict.pt"

  -- load example image data
  S.IVGenericDict example <- pickleLoad "mnist.example.pt"
  let ivt = snd $ example !! 0
  let S.IVTensor t = ivt

  -- load torchscript module
  tsModule <- S.loadScript WithoutRequiredGrad "mnist.ts.pt"

  -- perform inference computation
  let result = forward tsModule [ivt]
  print result

  putStrLn "Done"
