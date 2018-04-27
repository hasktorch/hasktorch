{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor where

import Data.Coerce (coerce)
import Data.Typeable
import Data.Maybe (fromMaybe)
import Data.List (intercalate)
import Data.List.NonEmpty (NonEmpty(..), toList)
import Control.Applicative ((<|>))
import Control.Monad
import System.IO.Unsafe
import Control.Monad.Managed
import Control.Exception.Safe
import Foreign.C.Types
import GHC.ForeignPtr (ForeignPtr)
import GHC.Int

import qualified Torch.Types.TH            as TH
import qualified Foreign.Marshal.Array     as FM
import qualified Torch.Sig.State           as Sig
import qualified Torch.Sig.Types           as Sig
import qualified Torch.Sig.Types.Global    as Sig
import qualified Torch.Sig.Tensor          as Sig
import qualified Torch.Sig.Tensor.Memory   as Sig
import qualified Torch.Sig.Storage         as StorageSig (c_size)

import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Internal
import Torch.Indef.Index

instance Show Dynamic where
  show t = unsafePerformIO $ do
    SomeDims ds <- getDims t
    (vs, desc) <- showTensor (get1d t) (get2d t) (get3d t) (get4d t) (dimVals ds)
    pure (vs ++ "\n" ++ desc)
  {-# NOINLINE show #-}


-- CPU ONLY:
--   desc :: Dynamic -> IO (DescBuff t)

_clearFlag :: Dynamic -> Int8 -> IO ()
_clearFlag t cc = withDynamicState t $ shuffle2 Sig.c_clearFlag (CChar cc)

tensordata :: Dynamic -> IO [HsReal]
tensordata t = withDynamicState t $ \s' t' ->
  ptrArray2hs (Sig.c_data s') (arrayLen s') (Sig.ctensor t)
 where
  arrayLen :: Ptr CState -> Ptr CTensor -> IO Int
  arrayLen s' p = Sig.c_storage s' p >>= fmap fromIntegral . StorageSig.c_size s'

get1d :: Dynamic -> Int64 -> IO HsReal
get1d t d1 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get1d s t' (fromIntegral d1)

get2d :: Dynamic -> Int64 -> Int64 -> IO HsReal
get2d t d1 d2 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get2d s t' (fromIntegral d1) (fromIntegral d2)

get3d :: Dynamic -> Int64 -> Int64 -> Int64 -> IO HsReal
get3d t d1 d2 d3 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get3d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3)

get4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal
get4d t d1 d2 d3 d4 = withDynamicState t $ \s t' -> c2hsReal <$> Sig.c_get4d s t' (fromIntegral d1) (fromIntegral d2) (fromIntegral d3) (fromIntegral d4)

isContiguous :: Dynamic -> IO Bool
isContiguous t = withDynamicState t $ \s t' ->
  (1 ==) <$> Sig.c_isContiguous s t'

isSameSizeAs :: Dynamic -> Dynamic -> IO Bool
isSameSizeAs t0 t1 = with2DynamicState t0 t1 $ \s t0' t1' ->
  (1 ==) <$> Sig.c_isSetTo s t0' t1'

isSetTo :: Dynamic -> Dynamic -> IO Bool
isSetTo t0 t1 = with2DynamicState t0 t1 $ \s t0' t1' ->
  (1 ==) <$> Sig.c_isSetTo s t0' t1'

isSize :: Dynamic -> TH.LongStorage -> IO Bool
isSize t ls = withDynamicState t $ \s t' ->
  withForeignPtr (snd $ TH.longStorageState ls) (fmap (1 ==) . Sig.c_isSize s t')

nDimension :: Dynamic -> IO Int32
nDimension t = withDynamicState t (\s t' -> fromIntegral <$> Sig.c_nDimension s t')

nElement :: Dynamic -> IO Int64
nElement t = withDynamicState t (\s t' -> fmap fromIntegral $ Sig.c_nElement s t')

_narrow :: Dynamic -> Dynamic -> DimVal -> Int64 -> Size -> IO ()
_narrow t0 t1 a b c = withDynamicState t0 $ \s t0' ->
  withForeignPtr (ctensor t1) $ \t1' ->
    Sig.c_narrow s t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

empty :: IO Dynamic
empty = Sig.newCState >>= \s -> Sig.c_new s >>= mkDynamic s

newExpand :: Dynamic -> TH.IndexStorage -> IO Dynamic
newExpand r ix = flip with pure $ do
  s <- manage' Sig.dynamicStateRef r
  r' <- manage' Sig.ctensor r
  ix' <- manage' (snd . TH.longStorageState) ix
  liftIO $ Sig.c_newExpand s r' ix' >>= mkDynamic s

_expand :: Dynamic -> Dynamic -> TH.IndexStorage -> IO ()
_expand r t ix = runManaged . joinIO $ Sig.c_expand
  <$> manage' Sig.dynamicStateRef r
  <*> manage' Sig.ctensor r
  <*> manage' Sig.ctensor t
  <*> manage' (snd . TH.longStorageState) ix

_expandNd  :: NonEmpty Dynamic -> NonEmpty Dynamic -> Int -> IO ()
_expandNd (rets@(s:|_)) ops i = runManaged $ do
  st    <- manage' Sig.dynamicStateRef s
  rets' <- mngNonEmpty rets
  ops'  <- mngNonEmpty ops
  liftIO $ Sig.c_expandNd st rets' ops' (fromIntegral i)
 where
  mngNonEmpty :: NonEmpty Dynamic -> Managed (Ptr (Ptr CTensor))
  mngNonEmpty = mapM toMPtr . toList >=> mWithArray

  mWithArray :: [Ptr a] -> Managed (Ptr (Ptr a))
  mWithArray as = managed (FM.withArray as)

  toMPtr :: Dynamic -> Managed (Ptr CTensor)
  toMPtr d = managed (withForeignPtr (Sig.ctensor d))


newClone :: Dynamic -> IO Dynamic
newClone t = withDynamicState t $ \s' t' -> Sig.c_newClone s' t' >>= mkDynamic s'

newContiguous :: Dynamic -> IO Dynamic
newContiguous t = withDynamicState t $ \s' t' -> Sig.c_newContiguous s' t' >>= mkDynamic s'

newNarrow :: Dynamic -> DimVal -> Int64 -> Size -> IO Dynamic
newNarrow t a b c = withDynamicState t $ \s' t' -> Sig.c_newNarrow s' t' (fromIntegral a) (fromIntegral b) (fromIntegral c) >>= mkDynamic s'

newSelect :: Dynamic -> DimVal -> Int64 -> IO Dynamic
newSelect t a b = withDynamicState t $ \s' t' -> Sig.c_newSelect s' t' (fromIntegral a) (fromIntegral b) >>= mkDynamic s'

newSizeOf :: Dynamic -> IO (TH.IndexStorage)
newSizeOf t = withDynamicState t $ \s' t' -> Sig.c_newSizeOf s' t' >>= mkCPUIxStorage

newStrideOf :: Dynamic -> IO (TH.IndexStorage)
newStrideOf t = withDynamicState t $ \s' t' -> Sig.c_newStrideOf s' t' >>= mkCPUIxStorage

newTranspose :: Dynamic -> DimVal -> DimVal -> IO Dynamic
newTranspose t a b = withDynamicState t $ \s' t' -> Sig.c_newTranspose s' t' (fromIntegral a) (fromIntegral b) >>= mkDynamic s'

newUnfold :: Dynamic -> DimVal -> Int64 -> Int64 -> IO Dynamic
newUnfold t a b c = withDynamicState t $ \s' t' -> Sig.c_newUnfold s' t' (fromIntegral a) (fromIntegral b) (fromIntegral c) >>= mkDynamic s'

newView :: Dynamic -> TH.IndexStorage -> IO Dynamic
newView t ix = withDynamicState t $ \s' t' ->
  withCPUIxStorage ix (Sig.c_newView s' t' >=> mkDynamic s')

newWithSize :: TH.IndexStorage -> TH.IndexStorage -> IO Dynamic
newWithSize l0 l1 =
  withCPUIxStorage l0 $ \l0' ->
    withCPUIxStorage l1 $ \l1' ->
      mkDynamicIO $ \s ->
        Sig.c_newWithSize s l0' l1'

newWithSize1d :: Size -> IO Dynamic
newWithSize1d a0 = mkDynamicIO $ \s -> Sig.c_newWithSize1d s (fromIntegral a0)

newWithSize2d :: Size -> Size -> IO Dynamic
newWithSize2d a0 a1 = mkDynamicIO $ \s -> Sig.c_newWithSize2d s (fromIntegral a0) (fromIntegral a1)

newWithSize3d :: Size -> Size -> Size -> IO Dynamic
newWithSize3d a0 a1 a2 = mkDynamicIO $ \s -> Sig.c_newWithSize3d s (fromIntegral a0) (fromIntegral a1) (fromIntegral a2)

newWithSize4d :: Size -> Size -> Size -> Size -> IO Dynamic
newWithSize4d a0 a1 a2 a3 = mkDynamicIO $ \s -> Sig.c_newWithSize4d s (fromIntegral a0) (fromIntegral a1) (fromIntegral a2) (fromIntegral a3)

newWithStorage :: Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> IO Dynamic
newWithStorage s pd l0 l1 =
  withStorageState s $ \state' s' ->
    withForeignPtr (snd $ TH.longStorageState l0) $ \l0' ->
      withForeignPtr (snd $ TH.longStorageState l1) $ \l1' ->
        Sig.c_newWithStorage state' s' (fromIntegral pd) l0' l1'
        >>= mkDynamic state'

newWithStorage1d :: Storage -> StorageOffset -> (Size, Stride) -> IO Dynamic
newWithStorage1d s pd (d00,d01) =
  withStorageState s $ \state' s' ->
    Sig.c_newWithStorage1d state' s' (fromIntegral pd)
    (fromIntegral d00) (fromIntegral d01)
    >>= mkDynamic state'


newWithStorage2d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
newWithStorage2d s pd (d00,d01) (d10,d11) =
  withStorageState s $ \state' s' ->
    Sig.c_newWithStorage2d state' s' (fromIntegral pd)
    (fromIntegral d00) (fromIntegral d01)
    (fromIntegral d10) (fromIntegral d11)
    >>= mkDynamic state'


newWithStorage3d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
newWithStorage3d s pd (d00,d01) (d10,d11) (d20,d21) =
  withStorageState s $ \state' s' ->
    Sig.c_newWithStorage3d state' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
    >>= mkDynamic state'


newWithStorage4d :: Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO Dynamic
newWithStorage4d s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
  withStorageState s $ \state' s' ->
    Sig.c_newWithStorage4d state' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
      (fromIntegral d30) (fromIntegral d31)
    >>= mkDynamic state'

newWithTensor :: Dynamic -> IO Dynamic
newWithTensor t = withDynamicState t $ \s' t' -> Sig.c_newWithTensor s' t' >>= mkDynamic s'

_resize :: Dynamic -> TH.IndexStorage -> TH.IndexStorage -> IO ()
_resize t l0 l1 = withDynamicState t $ \s' t' -> runManaged $ do
  l0' <- managed $ withCPUIxStorage l0
  l1' <- managed $ withCPUIxStorage l1
  liftIO $ Sig.c_resize s' t' l0' l1'

_resize1d :: Dynamic -> Int64 -> IO ()
_resize1d t l0 = withDynamicState t (\s' t' -> Sig.c_resize1d s' t' (fromIntegral l0))

_resize2d :: Dynamic -> Int64 -> Int64 -> IO ()
_resize2d t l0 l1 = withDynamicState t $ \s' t' -> Sig.c_resize2d s' t'
    (fromIntegral l0) (fromIntegral l1)

_resize3d :: Dynamic -> Int64 -> Int64 -> Int64 -> IO ()
_resize3d t l0 l1 l2 = withDynamicState t $ \s' t' -> Sig.c_resize3d s' t'
    (fromIntegral l0) (fromIntegral l1) (fromIntegral l2)

_resize4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
_resize4d t l0 l1 l2 l3 = withDynamicState t $ \s' t' -> Sig.c_resize4d s' t'
    (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3)

_resize5d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
_resize5d t l0 l1 l2 l3 l4 = withDynamicState t $ \s' t' -> Sig.c_resize5d s' t'
    (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (fromIntegral l4)

_resizeAs :: Dynamic -> Dynamic -> IO ()
_resizeAs t0 t1 = with2DynamicState t0 t1 Sig.c_resizeAs

_resizeNd :: Dynamic -> Int32 -> [Size] -> [Stride] -> IO ()
_resizeNd t i l0' l1' = do
  l0 <- FM.newArray (coerce l0' :: [CLLong])
  l1 <- FM.newArray (coerce l1' :: [CLLong])
  withDynamicState t $ \s' t' -> Sig.c_resizeNd s' t' (fromIntegral i) l0 l1

retain :: Dynamic -> IO ()
retain t = withDynamicState t Sig.c_retain

_select :: Dynamic -> Dynamic -> DimVal -> Int64 -> IO ()
_select t0 t1 a b = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_select s' t0' t1' (fromIntegral a) (fromIntegral b)

_set :: Dynamic -> Dynamic -> IO ()
_set t0 t1 = with2DynamicState t0 t1 Sig.c_set

_set1d :: Dynamic -> Int64 -> HsReal -> IO ()
_set1d t l0 v = withDynamicState t $ \s' t' -> Sig.c_set1d s' t' (fromIntegral l0) (hs2cReal v)

_set2d :: Dynamic -> Int64 -> Int64 -> HsReal -> IO ()
_set2d t l0 l1 v = withDynamicState t $ \s' t' -> Sig.c_set2d s' t' (fromIntegral l0) (fromIntegral l1) (hs2cReal v)

_set3d :: Dynamic -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
_set3d t l0 l1 l2 v = withDynamicState t $ \s' t' -> Sig.c_set3d s' t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (hs2cReal v)

_set4d :: Dynamic -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal -> IO ()
_set4d t l0 l1 l2 l3 v = withDynamicState t $ \s' t' -> Sig.c_set4d s' t' (fromIntegral l0) (fromIntegral l1) (fromIntegral l2) (fromIntegral l3) (hs2cReal v)

_setFlag :: Dynamic -> Int8 -> IO ()
_setFlag t l0 = withDynamicState t $ shuffle2 Sig.c_setFlag (CChar l0)

_setStorage :: Dynamic -> Storage -> StorageOffset -> TH.IndexStorage -> TH.IndexStorage -> IO ()
_setStorage t s a b c = withDynamicStateAndStorage t s $ \st' t' s' ->
  runManaged $ do
    b' <- managed $ withCPUIxStorage b
    c' <- managed $ withCPUIxStorage c
    liftIO $ Sig.c_setStorage st' t' s' (fromIntegral a) b' c'

_setStorage1d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> IO ()
_setStorage1d t s pd (d00,d01) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage1d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)

_setStorage2d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
_setStorage2d t s pd (d00,d01) (d10,d11) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage2d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)


_setStorage3d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
_setStorage3d t s pd (d00,d01) (d10,d11) (d20,d21) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage3d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)

_setStorage4d :: Dynamic -> Storage -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
_setStorage4d t s pd (d00,d01) (d10,d11) (d20,d21) (d30,d31) =
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorage4d st' t' s' (fromIntegral pd)
      (fromIntegral d00) (fromIntegral d01)
      (fromIntegral d10) (fromIntegral d11)
      (fromIntegral d20) (fromIntegral d21)
      (fromIntegral d30) (fromIntegral d31)

_setStorageNd :: Dynamic -> Storage -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
_setStorageNd t s a b hsc hsd = do
  c <- FM.newArray (coerce hsc :: [CLLong])
  d <- FM.newArray (coerce hsd :: [CLLong])
  withDynamicStateAndStorage t s $ \st' t' s' ->
    Sig.c_setStorageNd st' t' s' (fromIntegral a) (fromIntegral b) c d

size :: Dynamic -> DimVal -> IO Size
size t l0 = withDynamicState t $ \st' t' -> fromIntegral <$> Sig.c_size st' t' (fromIntegral l0)

sizeDesc :: Dynamic -> IO DescBuff
sizeDesc t = withDynamicState t $ \s' t' -> Sig.c_sizeDesc s' t' >>= Sig.descBuff

_squeeze :: Dynamic -> Dynamic -> IO ()
_squeeze t0 t1 = with2DynamicState t0 t1 Sig.c_squeeze

_squeeze1d :: Dynamic -> Dynamic -> DimVal -> IO ()
_squeeze1d t0 t1 d = with2DynamicState t0 t1 (shuffle3 Sig.c_squeeze1d (fromIntegral d))

storage :: Dynamic -> IO Storage
storage t = withDynamicState t $ \s' t' -> Sig.c_storage s' t' >>= mkStorage s'

storageOffset :: Dynamic -> IO StorageOffset
storageOffset t = withDynamicState t (fmap fromIntegral .: Sig.c_storageOffset)

stride :: Dynamic -> DimVal -> IO Stride
stride t a = withDynamicState t (fmap fromIntegral .: shuffle2 Sig.c_stride (fromIntegral a))

_transpose :: Dynamic -> Dynamic -> DimVal -> DimVal -> IO ()
_transpose t0 t1 a b = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_transpose s' t0' t1' (fromIntegral a) (fromIntegral b)

_unfold :: Dynamic -> Dynamic -> DimVal -> Size -> Step -> IO ()
_unfold t0 t1 a b c = with2DynamicState t0 t1 $ \s' t0' t1' ->
  Sig.c_unfold s' t0' t1' (fromIntegral a) (fromIntegral b) (fromIntegral c)

_unsqueeze1d :: Dynamic -> Dynamic -> DimVal -> IO ()
_unsqueeze1d t0 t1 d = with2DynamicState t0 t1 $
  shuffle3 Sig.c_unsqueeze1d (fromIntegral d)

-- ========================================================================= --
-- User API (can be bundled into the above)
-- ========================================================================= --

shape :: Dynamic -> IO [Size]
shape t = do
  ds <- nDimension t
  mapM (size t . fromIntegral) [0..ds-1]

-- not actually "inplace" this is actually "with return and static dimensions"
withInplace :: (Dynamic -> IO ()) -> Dim (d::[Nat]) -> IO Dynamic
withInplace op d = new d >>= \r -> op r >> pure r

-- not actually "inplace" this is actually "with return and runtime dimensions"
withInplace' :: (Dynamic -> IO ()) -> SomeDims -> IO Dynamic
withInplace' op (SomeDims d) = withInplace op d

-- This is actually 'inplace'
twice :: Dynamic -> (Dynamic -> Dynamic -> IO ()) -> IO Dynamic
twice t op = op t t >> pure t

-- Should be renamed to @newFromSize@
withEmpty :: Dynamic -> (Dynamic -> IO ()) -> IO Dynamic
withEmpty t op = getDims t >>= new' >>= \r -> op r >> pure r

-- We can get away with this some of the time, when Torch does the resizing in C, but you need to look at
-- the c implementation
withEmpty' :: (Dynamic -> IO ()) -> IO Dynamic
withEmpty' op = empty >>= \r -> op r >> pure r


_setStorageDim :: Dynamic -> Storage -> StorageOffset -> [(Size, Stride)] -> IO ()
_setStorageDim t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> _setStorage1d t s o x
  [x, y]       -> _setStorage2d t s o x y
  [x, y, z]    -> _setStorage3d t s o x y z
  [x, y, z, q] -> _setStorage4d t s o x y z q
  _            -> throwGT4 "setStorage"

_setDim :: Dynamic -> Dim (d::[Nat]) -> HsReal -> IO ()
_setDim t d v = case dimVals d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> _set1d t x       v
  [x, y]       -> _set2d t x y     v
  [x, y, z]    -> _set3d t x y z   v
  [x, y, z, q] -> _set4d t x y z q v
  _            -> throwGT4 "set"

_resizeDim :: Dynamic -> Dim (d::[Nat]) -> IO ()
_resizeDim t d = case dimVals d of
  []              -> throwNE "can't resize to an empty dimension."
  [x]             -> _resize1d t x
  [x, y]          -> _resize2d t x y
  [x, y, z]       -> _resize3d t x y z
  [x, y, z, q]    -> _resize4d t x y z q
  [x, y, z, q, w] -> _resize5d t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
  -- ds              -> _resizeNd t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")

-- FIXME construct this with TH, not with the setting, which might be doing a second linear pass
fromList1d :: [HsReal] -> IO Dynamic
fromList1d l = do
  res <- new' =<< someDimsM [length l]
  mapM_  (upd res) (zip [0..length l - 1] l)
  pure res
 where
  upd :: Dynamic -> (Int, HsReal) -> IO ()
  upd t (idx, v) = someDimsM [idx] >>= \sd -> setDim'_ t sd v

resizeDim :: Dynamic -> Dim (d::[Nat]) -> IO Dynamic
resizeDim src d = newClone src >>= \res -> _resizeDim res d >> pure res

resizeDim' :: Dynamic -> SomeDims -> IO Dynamic
resizeDim' t (SomeDims d) = resizeDim t d

getDim :: Dynamic -> Dim (d::[Nat]) -> IO HsReal
getDim t d = case dimVals d of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> throwGT4 "get"

getDims :: Dynamic -> IO SomeDims
getDims = getDimList >=> someDimsM

getDimList :: Dynamic -> IO [Size]
getDimList t = do
  nd <- nDimension t
  mapM (size t . fromIntegral) [0 .. nd -1]

new :: Dim (d::[Nat]) -> IO Dynamic
new d = case dimVals d of
  []           -> empty
  [x]          -> newWithSize1d x
  [x, y]       -> newWithSize2d x y
  [x, y, z]    -> newWithSize3d x y z
  [x, y, z, q] -> newWithSize4d x y z q
  _ -> do
    t <- empty
    _resizeDim t d
    pure t

setDim'_ :: Dynamic -> SomeDims -> HsReal -> IO ()
setDim'_ t (SomeDims d) v = _setDim t d v

resizeDim'_ :: Dynamic -> SomeDims -> IO ()
resizeDim'_ t (SomeDims d) = _resizeDim t d

getDim' :: Dynamic -> SomeDims -> IO HsReal
getDim' t (SomeDims d) = getDim t d

new' :: SomeDims -> IO Dynamic
new' (SomeDims d) = new d

-- Is this right? why are there three tensors
resizeAs :: Dynamic -> Dynamic -> IO Dynamic
resizeAs src shape = do
  res <- newClone src
  _resizeAs res shape
  pure res

showTensor
  :: forall a . (Typeable a, Ord a, Num a, Show a)
  => (Int64 -> IO a)
  -> (Int64 -> Int64 -> IO a)
  -> (Int64 -> Int64 -> Int64 -> IO a)
  -> (Int64 -> Int64 -> Int64 -> Int64 -> IO a)
  -> [Int64]
  -> IO (String, String)
showTensor get'1d get'2d get'3d get'4d ds =
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
