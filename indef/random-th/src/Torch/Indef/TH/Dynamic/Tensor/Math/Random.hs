module Torch.Indef.TH.Dynamic.Tensor.Math.Random where

import Torch.Indef.Types
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Sig.TH.Tensor.Math.Random as Sig
import qualified Torch.Class.TH.Tensor.Math.Random as Class

go
  :: (Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr TH.CLongStorage -> IO ())
  -> Dynamic
  -> Generator
  -> TH.IndexStorage
  -> IO ()
go fn d g i = runManaged . joinIO $ fn
  <$> manage' Sig.dynamicStateRef d
  <*> manage' Sig.ctensor d
  <*> manage' Sig.rng g
  <*> manage' (snd . TH.longStorageState) i

instance Class.THTensorMathRandom Dynamic where
  rand_  :: Dynamic -> Generator -> TH.IndexStorage -> IO ()
  rand_ = go Sig.c_rand

  randn_  :: Dynamic -> Generator -> TH.IndexStorage -> IO ()
  randn_ = go Sig.c_randn

  randperm_ :: Dynamic -> Generator -> Integer -> IO ()
  randperm_ t g i = runManaged . joinIO $ Sig.c_randperm
    <$> manage' Sig.dynamicStateRef t
    <*> manage' Sig.ctensor t
    <*> manage' Sig.rng g
    <*> pure (fromIntegral i)


