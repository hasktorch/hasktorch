{- O-P-T-I-O-N-S_GHC -F -pgmF hspec-discover -}
module Main where

import Test.Hspec
import qualified MemorySpec as MS
import qualified RawLapackSVDSpec as SVDS
import qualified GarbageCollectionSpec as GS
import qualified Torch.Core.LogAddSpec as LS
import qualified Torch.Core.RandomSpec as RS
import qualified Torch.Static.NN.AbsSpec as AbsNN
import qualified Torch.Static.NN.LinearSpec as LinearNN

main :: IO ()
main = hspec $ do
  describe "MemorySpec" MS.spec
  describe "RawLapackSVDSpec" SVDS.spec
  describe "GarbageCollectionSpec" GS.spec
  describe "Torch.Core.LogAddSpec" LS.spec
  describe "Torch.Core.RandomSpec" RS.spec
  describe "Torch.Static.NN.AbsSpec" AbsNN.spec
  describe "Torch.Static.NN.LinearSpec" LinearNN.spec


