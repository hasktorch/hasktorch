-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Index
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Redundant version of @Torch.Indef.{Dynamic/Static}.Tensor@ for Index tensors.
--
-- FIXME: in the future, there could be a smaller subset of Torch which could
-- be compiled to to keep the code dry. Alternatively, if backpack one day
-- supports recursive indefinites, we could use this feature to possibly remove
-- this package and 'Torch.Indef.Mask'.
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE MonoLocalBinds #-}
module Torch.Indef.Index
  ( newIx
  , newIxDyn
  , zeroIxNd
  , index
  , index1d
  , indexNd
  , indexDyn
  , mkCPUIx
  , withCPUIxStorage
  , withIxStorage
  , withDynamicState
  , mkCPUIxStorage
  , ixShape
  , ixCPUStorage
  , showIx
  ) where

import Foreign
import Foreign.Ptr
import Data.Proxy
import Data.List
import Data.Typeable
import Data.Maybe
import Data.Singletons.Prelude.List
import Control.Monad
import Control.Applicative
import System.IO.Unsafe
import Numeric.Dimensions

import Torch.Sig.State as Sig
import Torch.Sig.Types.Global as Sig
import Torch.Indef.Types hiding (withDynamicState, mkDynamic, mkDynamicIO)
import Torch.Indef.Internal
import qualified Torch.Types.TH as TH
import qualified Torch.Types.TH.Long as THLong
import qualified Torch.Sig.Index.Tensor as IxSig
import qualified Torch.Sig.Index.TensorFree as IxSig
import qualified Torch.FFI.TH.Long.Storage as LongStorage
import qualified Torch.FFI.TH.Long.Storage as LongStorage
import qualified Torch.FFI.TH.Long.Tensor as LongTensor
import qualified Foreign as FM
import qualified Foreign.Marshal.Array as FM

-- | build a new static index tensor
--
-- FIXME: This can abstracted away with backpack, but I'm not sure if it can do it atm.
newIx :: forall n . KnownDim n => IndexTensor '[n]
newIx = longAsStatic $ newIxDyn (dimVal (dim :: Dim n))

-- | build a new, empty, static index tensor with no term-level dimensions -- but allow
-- the type-level dimensions to vary.
--
-- FIXME: this is a bad function and should be replaced with a 'Torch.Indef.Static.Tensor.new' function.
zeroIxNd :: Dimensions d => IndexTensor d
zeroIxNd = longAsStatic $ newIxDyn 0

-- | build a new 1-dimensional, dynamically-typed index tensor of lenght @i@
newIxDyn :: Integral i => i -> IndexDynamic
newIxDyn x = unsafeDupablePerformIO $
  withForeignPtr Sig.torchstate $ \s ->
    IxSig.c_newWithSize1d s (fromIntegral x) >>= mkDynamic

-- | Make a dynamic, 1d index tensor from a list.
--
-- FIXME construct this with TH, not with the setting, which might be doing a second linear pass
indexDyn :: [Integer] -> IndexDynamic
indexDyn l = unsafeDupablePerformIO $ do
  let res = newIxDyn (length l)
  mapM_  (upd res) (zip [0..length l - 1] l)
  pure res

  where
    upd :: IndexDynamic -> (Int, Integer) -> IO ()
    upd t (idx, v) = withDynamicState t $ \s' t' -> IxSig.c_set1d s' t' (fromIntegral idx) (fromIntegral v)

-- | purely make a 1d static index tensor from a list of integers. Returns Nothing if the
-- list does not match the expected size of the tensor.
--
-- should be depreciated in favor of 'index1d'
index :: forall n . KnownDim n => [Integer] -> Maybe (IndexTensor '[n])
index l
  | genericLength l == dimVal (dim :: Dim n) = Just . longAsStatic . indexDyn $ l
  | otherwise = Nothing

-- | alias to 'index' and should subsume it when this package no longer assumes that index tensors are 1d.
index1d :: KnownDim n => [Integer] -> Maybe (IndexTensor '[n])
index1d = index

-- | n-dimensional version of 'index1d'.
--
-- FIXME: this relies on 'indexDyn' which only makes 1d tensors.
indexNd :: forall d . KnownDim (Product d) => [Integer] -> Maybe (IndexTensor d)
indexNd l
  | genericLength l == dimVal (dim :: Dim (Product d)) = Just . longAsStatic . indexDyn $ l
  | otherwise = Nothing

-- | Convenience method for 'newWithData' specific to longs for making CPU Long storage.
ixCPUStorage :: [Integer] -> IO TH.LongStorage
ixCPUStorage pr = withForeignPtr TH.torchstate $ \st -> do
  pr' <- FM.withArray (THLong.hs2cReal <$> pr) pure
  thl <- LongStorage.c_newWithData st pr' (fromIntegral $ length pr)
  TH.LongStorage <$> ((TH.torchstate,)
    <$> FM.newForeignPtr LongStorage.p_free thl)

-- | resize a 1d dynamic index tensor.
--
-- FIXME: export or remove this function as appropriate.
_resizeDim1d :: IndexDynamic -> Integer -> IO ()
_resizeDim1d t x = withDynamicState t $ \s' t' -> IxSig.c_resize1d s' t' (fromIntegral x)

-- | make a dynamic CPU tensor from a raw torch ctensor
mkCPUIx :: Ptr TH.C'THLongTensor -> IO TH.LongDynamic
mkCPUIx p = fmap TH.LongDynamic
  $ (TH.torchstate,)
  <$> newForeignPtr LongTensor.p_free p

-- | run a function with access to a raw CPU-bound Long tensor storage.
withCPUIxStorage :: TH.LongStorage -> (Ptr TH.C'THLongStorage -> IO x) -> IO x
withCPUIxStorage ix fn = withForeignPtr (snd $ TH.longStorageState ix) fn

-- | run a function with access to a dynamic index tensor's raw internal state and c-pointer.
withDynamicState :: IndexDynamic -> (Ptr Sig.CState -> Ptr Sig.CLongTensor -> IO x) -> IO x
withDynamicState t fn = do
  withForeignPtr (fst $ Sig.longDynamicState t) $ \sref ->
    withForeignPtr (snd $ Sig.longDynamicState t) $ \tref ->
      fn sref tref

-- | run a function with access to a dynamic index storage's raw internal state and c-pointer.
withIxStorage :: Sig.IndexStorage -> (Ptr CLongStorage -> IO x) -> IO x
withIxStorage ix fn = withForeignPtr (snd $ Sig.longStorageState ix) fn

-- | make a dynamic CPU tensor's storage from a raw torch LongStorage
mkCPUIxStorage :: Ptr TH.C'THLongStorage -> IO TH.LongStorage
mkCPUIxStorage p = fmap TH.LongStorage
  $ (TH.torchstate,)
  <$> newForeignPtr LongStorage.p_free p

-- | get the shape of a static index tensor from the term-level
ixShape :: IndexTensor d -> [Word]
ixShape t = unsafeDupablePerformIO $ withDynamicState (longAsDynamic t) $ \s' t' -> do
  ds <- IxSig.c_nDimension s' t'
  mapM (fmap fromIntegral . IxSig.c_size s' t' . fromIntegral) [0..ds-1]

-- | show an index.
--
-- FIXME: because we are using backpack, we can't declare a show instance on the IndexTensor both
-- here and in the signatures. /if we want this functionality we must operate on raw code and write
-- the show instance in hasktorch-types/.
{-# NOINLINE showIx #-}
showIx t = unsafePerformIO $ do
  let ds = fromIntegral <$> ixShape t
  (vs, desc) <- go (ixGet1d t) (ixGet2d t) (ixGet3d t) (ixGet4d t) ds
  pure (vs ++ "\n" ++ desc)
 where
  ixGet1d :: IndexTensor d -> Int64 -> IO Integer
  ixGet1d it i = fmap fromIntegral . withDynamicState (longAsDynamic it) $ \s' it' -> IxSig.c_get1d s' it'
    (fromIntegral i)
  ixGet2d :: IndexTensor d -> Int64 -> Int64 -> IO Integer
  ixGet2d it i i1 = fmap fromIntegral . withDynamicState (longAsDynamic it) $ \s' it' -> IxSig.c_get2d s' it'
    (fromIntegral i) (fromIntegral i1)
  ixGet3d :: IndexTensor d -> Int64 -> Int64 -> Int64 -> IO Integer
  ixGet3d it i i1 i2 = fmap fromIntegral . withDynamicState (longAsDynamic it) $ \s' it' -> IxSig.c_get3d s' it'
    (fromIntegral i) (fromIntegral i1) (fromIntegral i2)

  ixGet4d :: IndexTensor d -> Int64 -> Int64 -> Int64 -> Int64 -> IO Integer
  ixGet4d it i i1 i2 i3 = fmap fromIntegral . withDynamicState (longAsDynamic it) $ \s' it' -> IxSig.c_get4d s' it'
    (fromIntegral i) (fromIntegral i1) (fromIntegral i2) (fromIntegral i3)

  go
    :: forall a . (Typeable a, Ord a, Num a, Show a)
    => (Int64 -> IO a)
    -> (Int64 -> Int64 -> IO a)
    -> (Int64 -> Int64 -> Int64 -> IO a)
    -> (Int64 -> Int64 -> Int64 -> Int64 -> IO a)
    -> [Int64]
    -> IO (String, String)
  go get'1d get'2d get'3d get'4d ds =
    (,desc) <$> case ds of
      []  -> pure ""
      [x] -> brackets . intercalate "" <$> mapM (fmap valWithSpace . get'1d) (mkIx x)
      [x,y] -> go "" get'2d x y
      [x,y,z] -> mat3dGo x y z
      [x,y,z,q] -> mat4dGo x y z q
      _ -> pure "Can't print this yet"
   where
    go :: String -> (Int64 -> Int64 -> IO a) -> Int64 -> Int64 -> IO String
    go fill getter x y = do
      vs <- mapM (fmap valWithSpace . uncurry getter) (mkXY x y)
      pure (mat2dGo fill y "" vs)

    mat2dGo :: String -> Int64 -> String -> [String] -> String
    mat2dGo    _ _ acc []  = acc
    mat2dGo fill y acc rcs = mat2dGo fill y acc' rest
      where
        (row, rest) = splitAt (fromIntegral y) rcs
        fullrow = fill ++ brackets (intercalate "" row)
        acc' = if null acc then fullrow else acc ++ "\n" ++ fullrow

    mat3dGo :: Int64 -> Int64 -> Int64 -> IO String
    mat3dGo x y z = fmap (intercalate "") $ forM (mkIx x) $ \x' -> do
      mat <- go "  " (get'3d x') y z
      pure $ gt2IxHeader [x'] ++ mat

    mat4dGo :: Int64 -> Int64 -> Int64 -> Int64 -> IO String
    mat4dGo w q x y = fmap (intercalate "") $ forM (mkXY w q) $ \(w', q') -> do
      mat <- go "  " (get'4d w' q') x y
      pure $ gt2IxHeader [w', q'] ++ mat

    gt2IxHeader :: [Int64] -> String
    gt2IxHeader is = "\n(" ++ intercalate "," (fmap show is) ++",.,.):\n"

    mkIx :: Int64 -> [Int64]
    mkIx x = [0..x - 1]

    mkXY :: Int64 -> Int64 -> [(Int64, Int64)]
    mkXY x y = [ (r, c) | r <- mkIx x, c <- mkIx y ]

    brackets :: String -> String
    brackets s = "[" ++ s ++ "]"

    valWithSpace :: (Typeable a, Ord a, Num a, Show a) => a -> String
    valWithSpace v = spacing ++ value ++ " "
     where
       truncTo :: (RealFrac x, Fractional x) => Int -> x -> x
       truncTo n f = fromInteger (round $ f * (10^n)) / (10.0^^n)

       value :: String
       value = fromMaybe (show v) $
             (show . truncTo 6 <$> (cast v :: Maybe Double))
         <|> (show . truncTo 6 <$> (cast v :: Maybe Float))

       spacing = case compare (signum v) 0 of
          LT -> " "
          _  -> "  "

    descType, descShape, desc :: String
    descType = show (typeRep (Proxy :: Proxy a)) ++ " tensor with "
    descShape = "shape: " ++ intercalate "x" (fmap show ds)
    desc = brackets $ descType ++ descShape

-------------------------------------------------------------------------------
-- Helper functions which mimic code from 'Torch.Indef.Types'

mkDynamic :: Ptr Sig.CLongTensor -> IO IndexDynamic
mkDynamic t =
  withForeignPtr Sig.torchstate $ \s ->
    Sig.longDynamic Sig.torchstate
      <$> newForeignPtrEnv IxSig.p_free s t


