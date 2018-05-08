{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Index
  ( CPUIndex
  , CPUIndexStorage
  , newIx
  , newIxDyn
  , index
  , indexDyn
  , mkCPUIx
  , withCPUIxStorage
  , mkCPUIxStorage
  , mkLongStorage
  ) where

import Foreign
import Foreign.Ptr
import Data.Proxy
import Data.List
import System.IO.Unsafe

import Torch.Dimensions
import Torch.Sig.State as Sig
import Torch.Sig.Types.Global as Sig
import Torch.Indef.Types hiding (withDynamicState, mkDynamic, mkDynamicIO)
import Torch.Indef.Internal
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Index.Tensor as IxSig
import qualified Torch.Sig.Index.TensorFree as IxSig
import qualified Torch.FFI.TH.Long.Storage as LongStorage
import qualified Torch.FFI.TH.Long.Tensor as LongTensor

type CPUIndex = TH.LongDynamic
type CPUIndexStorage = TH.LongStorage

-- FIXME: This can abstracted away with backpack, but I'm not sure if it can do it atm.
newIx :: forall n . KnownNat n => IndexTensor '[n]
newIx = longAsStatic $ newIxDyn (natVal (Proxy :: Proxy n))

newIxDyn :: Integral i => i -> IndexDynamic
newIxDyn x = unsafeDupablePerformIO . mkDynamicIO $ \s ->
  IxSig.c_newWithSize1d s (fromIntegral x)

-- FIXME construct this with TH, not with the setting, which might be doing a second linear pass
indexDyn :: [Integer] -> IndexDynamic
indexDyn l = unsafeDupablePerformIO $ do
  let res = newIxDyn (length l)
  mapM_  (upd res) (zip [0..length l - 1] l)
  pure res

  where
    upd :: IndexDynamic -> (Int, Integer) -> IO ()
    upd t (idx, v) = withDynamicState t $ \s' t' -> IxSig.c_set1d s' t' (fromIntegral idx) (fromIntegral v)

index :: forall n . KnownNat n => [Integer] -> Maybe (IndexTensor '[n])
index l
  | genericLength l == natVal (Proxy :: Proxy n) = Just . longAsStatic . indexDyn $ l
  | otherwise = Nothing

_resizeDim :: IndexDynamic -> Integer -> IO ()
_resizeDim t x = withDynamicState t $ \s' t' -> IxSig.c_resize1d s' t' (fromIntegral x)

withDynamicState :: IndexDynamic -> (Ptr Sig.CState -> Ptr Sig.CLongTensor -> IO x) -> IO x
withDynamicState t fn = do
  withForeignPtr (fst $ Sig.longDynamicState t) $ \sref ->
    withForeignPtr (snd $ Sig.longDynamicState t) $ \tref ->
      fn sref tref

mkCPUIx :: Ptr TH.C'THLongTensor -> IO CPUIndex
mkCPUIx p = fmap TH.LongDynamic
  $ (,)
  <$> (TH.newCState >>= TH.manageState)
  <*> newForeignPtr LongTensor.p_free p

withCPUIxStorage :: CPUIndexStorage -> (Ptr TH.C'THLongStorage -> IO x) -> IO x
withCPUIxStorage ix fn = withForeignPtr (snd $ TH.longStorageState ix) fn

mkCPUIxStorage :: Ptr TH.C'THLongStorage -> IO CPUIndexStorage
mkCPUIxStorage p = fmap TH.LongStorage
  $ (,)
  <$> (TH.newCState >>= TH.manageState)
  <*> newForeignPtr LongStorage.p_free p

mkLongStorage :: Ptr TH.CLongStorage -> IO TH.LongStorage
mkLongStorage p = do
  fpState <- TH.newCState >>= TH.manageState
  fp <- newForeignPtr LongStorage.p_free p
  pure $ TH.LongStorage (fpState, fp)

mkDynamic :: Ptr Sig.CState -> Ptr Sig.CLongTensor -> IO IndexDynamic
mkDynamic s t = Sig.longDynamic
  <$> Sig.manageState s
  <*> newForeignPtrEnv IxSig.p_free s t

mkDynamicIO :: (Ptr Sig.CState -> IO (Ptr Sig.CLongTensor)) -> IO IndexDynamic
mkDynamicIO builder = Sig.newCState >>= \s ->
  builder s >>= mkDynamic s



