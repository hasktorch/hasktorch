

{-# LANGUAGE OverloadedStrings #-}
import Torch
import Torch.Script as S

loadFile file = do
  print file
  S.IVGenericDict vals <- pickleLoad file
  print vals
  pure vals

main = do
  let modelFile = "test2sd.pt"
  vals <- loadFile modelFile
  let v1 = vals !! 0
  let (IVString s1, IVTensor t1) = v1
  print vals
  print v1
  print s1
  print t1
  init <- sample $ LinearSpec { in_features = 2, out_features = 1 }
  -- loaded <- loadParams init "test2sd.pt"
  print init
  -- print loaded

  putStrLn "Done"
