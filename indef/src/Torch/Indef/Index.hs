{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TupleSections #-}
module Torch.Indef.Index
  ( CPUIndex
  , CPUIndexStorage
  , newIx
  , newIxDyn
  , zeroIxNd
  , index
  , index1d
  , indexNd
  , indexDyn
  , mkCPUIx
  , withCPUIxStorage
  , withDynamicState
  , mkCPUIxStorage
  , mkLongStorage
  , ixShape
  , ixGet1d
  , ixCPUStorage
  , showIx
  ) where


import Foreign
import Foreign.Ptr
import Data.Proxy
import Data.List
import Data.Typeable
import Control.Monad
import Control.Applicative
import Data.Maybe
import System.IO.Unsafe

import Torch.Dimensions
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

type CPUIndex = TH.LongDynamic
type CPUIndexStorage = TH.LongStorage

-- FIXME: This can abstracted away with backpack, but I'm not sure if it can do it atm.
newIx :: forall n . KnownDim n => IndexTensor '[n]
newIx = longAsStatic $ newIxDyn (dimVal (dim :: Dim n))

zeroIxNd :: Dimensions d => IndexTensor d
zeroIxNd = longAsStatic $ newIxDyn 0

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

index :: forall n . KnownDim n => [Integer] -> Maybe (IndexTensor '[n])
index l
  | genericLength l == dimVal (dim :: Dim n) = Just . longAsStatic . indexDyn $ l
  | otherwise = Nothing

index1d :: KnownDim n => [Integer] -> Maybe (IndexTensor '[n])
index1d = index

indexNd :: forall d . KnownDim (Product d) => Dimensions d => [Integer] -> Maybe (IndexTensor d)
indexNd l
  | genericLength l == dimVal (dim :: Dim (Product d)) = Just . longAsStatic . indexDyn $ l
  | otherwise = Nothing

-- | Convenience method for 'newWithData' specific to longs.
ixCPUStorage :: [Integer] -> IO TH.LongStorage
ixCPUStorage pr = do
  st  <- TH.newCState
  pr' <- FM.withArray (THLong.hs2cReal <$> pr) pure
  thl <- LongStorage.c_newWithData st pr' (fromIntegral $ length pr)
  TH.LongStorage <$> ((,)
    <$> TH.manageState st
    <*> FM.newForeignPtr LongStorage.p_free thl)

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

ixShape :: IndexTensor d -> IO [Word]
ixShape t = withDynamicState (longAsDynamic t) $ \s' t' -> do
  ds <- IxSig.c_nDimension s' t'
  mapM (fmap fromIntegral . IxSig.c_size s' t' . fromIntegral) [0..ds-1]

getDims :: IndexTensor d -> IO SomeDims
getDims = fmap someDimsVal . ixShape

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

{-# NOINLINE showIx #-}
showIx t = unsafePerformIO $ do
  SomeDims ds <- getDims t
  (vs, desc) <- go (ixGet1d t) (ixGet2d t) (ixGet3d t) (ixGet4d t) (fromIntegral <$> listDims ds)
  pure (vs ++ "\n" ++ desc)
 where
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


