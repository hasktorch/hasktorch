{-# LANGUAGE OverloadedStrings #-}

import Torch
import Torch.Script as S

main = do
  S.IVGenericDict vals <- pickleLoad "simple.dict.pt"
  let (_, IVTensor weight) = vals !! 0
  let (_, IVTensor bias) = vals !! 1
  weight' <- makeIndependent weight
  bias' <- makeIndependent bias
  let model = Linear weight' bias'
  let x = asTensor [1.0 :: Float, 2.0]
  print model
  print $ linear model x
