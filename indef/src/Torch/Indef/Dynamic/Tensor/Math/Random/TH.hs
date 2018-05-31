module Torch.Indef.Dynamic.Tensor.Math.Random.TH
  ( _rand
  , _randn
  , _randperm
  ) where

import Torch.Indef.Types
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Sig.Tensor.Math.Random.TH as Sig

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

_rand  :: Dynamic -> Generator -> TH.IndexStorage -> IO ()
_rand = go Sig.c_rand

_randn  :: Dynamic -> Generator -> TH.IndexStorage -> IO ()
_randn = go Sig.c_randn

_randperm :: Dynamic -> Generator -> Integer -> IO ()
_randperm t g i = runManaged . joinIO $ Sig.c_randperm
  <$> manage' Sig.dynamicStateRef t
  <*> manage' Sig.ctensor t
  <*> manage' Sig.rng g
  <*> pure (fromIntegral i)


