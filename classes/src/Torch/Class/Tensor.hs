{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Class.Tensor where

import Control.Arrow ((***))
import Control.Exception.Safe
import Control.Monad ((>=>), forM_)
import Control.Monad.IO.Class
import Data.List (genericLength)
import GHC.Int
import Torch.Class.Types
import Torch.Dimensions
import Data.List.NonEmpty (NonEmpty)
import qualified Torch.Types.TH as TH

class IsTensor t where
  _clearFlag :: t -> Int8 -> IO ()
  tensordata :: t -> IO [HsReal t]
  _free :: t -> IO ()
  _freeCopyTo :: t -> t -> IO ()
  get1d :: t -> Int64 -> IO (HsReal t)
  get2d :: t -> Int64 -> Int64 -> IO (HsReal t)
  get3d :: t -> Int64 -> Int64 -> Int64 -> IO (HsReal t)
  get4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO (HsReal t)
  isContiguous :: t -> IO Bool
  isSameSizeAs :: t -> t -> IO Bool
  isSetTo :: t -> t -> IO Bool
  isSize :: t -> TH.IndexStorage -> IO Bool
  nDimension :: t -> IO Int32
  nElement :: t -> IO Int64
  _narrow :: t -> t -> DimVal -> Int64 -> Size -> IO ()

  -- | renamed from TH's @new@ because this always returns an empty tensor
  empty :: IO t

  newExpand :: t -> TH.IndexStorage -> IO t
  expand :: t -> t -> TH.IndexStorage -> IO ()
  expandNd :: NonEmpty t -> NonEmpty t -> Int -> IO ()

  newClone :: t -> IO t
  newContiguous :: t -> IO t
  newNarrow :: t -> DimVal -> Int64 -> Size -> IO t
  newSelect :: t -> DimVal -> Int64 -> IO t
  newSizeOf :: t -> IO (TH.IndexStorage)
  newStrideOf :: t -> IO (TH.IndexStorage)
  newTranspose :: t -> DimVal -> DimVal -> IO t
  newUnfold :: t -> DimVal -> Int64 -> Int64 -> IO t
  newView :: t -> SizesStorage t -> IO t
  newWithSize :: SizesStorage t -> StridesStorage t -> IO t
  newWithSize1d :: Size -> IO t
  newWithSize2d :: Size -> Size -> IO t
  newWithSize3d :: Size -> Size -> Size -> IO t
  newWithSize4d :: Size -> Size -> Size -> Size -> IO t
  newWithStorage :: HsStorage t -> StorageOffset -> SizesStorage t -> StridesStorage t -> IO t
  newWithStorage1d :: HsStorage t -> StorageOffset -> (Size, Stride) -> IO t
  newWithStorage2d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO t
  newWithStorage3d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO t
  newWithStorage4d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO t
  newWithTensor :: t -> IO t
  _resize :: t -> SizesStorage t -> StridesStorage t -> IO ()
  _resize1d :: t -> Int64 -> IO ()
  _resize2d :: t -> Int64 -> Int64 -> IO ()
  _resize3d :: t -> Int64 -> Int64 -> Int64 -> IO ()
  _resize4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  _resize5d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  _resizeAs :: t -> t -> IO ()
  _resizeNd :: t -> Int32 -> [Size] -> [Stride] -> IO ()
  retain :: t -> IO ()
  _select :: t -> t -> DimVal -> Int64 -> IO ()
  _set :: t -> t -> IO ()
  _set1d :: t -> Int64 -> HsReal t -> IO ()
  _set2d :: t -> Int64 -> Int64 -> HsReal t -> IO ()
  _set3d :: t -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  _set4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  _setFlag :: t -> Int8 -> IO ()
  _setStorage :: t -> HsStorage t -> StorageOffset -> SizesStorage t -> StridesStorage t -> IO ()
  _setStorage1d :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> IO ()
  _setStorage2d :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  _setStorage3d :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  _setStorage4d :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  _setStorageNd :: t -> HsStorage t -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  size :: t -> DimVal -> IO Size
  sizeDesc :: t -> IO (DescBuff t)
  _squeeze :: t -> t -> IO ()
  _squeeze1d :: t -> t -> DimVal -> IO ()
  storage :: t -> IO (HsStorage t)
  storageOffset :: t -> IO StorageOffset
  stride :: t -> DimVal -> IO Stride
  _transpose :: t -> t -> DimVal -> DimVal -> IO ()
  _unfold :: t -> t -> DimVal -> Size -> Step -> IO ()
  _unsqueeze1d :: t -> t -> DimVal -> IO ()

class CPUTensor t where
  desc :: t -> IO (DescBuff t)

shape :: IsTensor t => t -> IO [Size]
shape t = do
  ds <- nDimension t
  mapM (size t . fromIntegral) [0..ds-1]

-- not actually "inplace" this is actually "with return and static dimensions"
withInplace :: IsTensor t => (t -> IO ()) -> Dim (d::[Nat]) -> IO t
withInplace op d = new d >>= \r -> op r >> pure r

-- not actually "inplace" this is actually "with return and runtime dimensions"
withInplace' :: IsTensor t => (t -> IO ()) -> SomeDims -> IO t
withInplace' op (SomeDims d) = withInplace op d

-- This is actually 'inplace'
twice :: IsTensor t => t -> (t -> t -> IO ()) -> IO t
twice t op = op t t >> pure t

-- I think we can get away with this for most creations. Torch does the resizing in C.
withEmpty :: IsTensor t => (t -> IO ()) -> IO t
withEmpty op = empty >>= \r -> op r >> pure r

_setStorageDim :: IsTensor t => t -> HsStorage t -> StorageOffset -> [(Size, Stride)] -> IO ()
_setStorageDim t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> _setStorage1d t s o x
  [x, y]       -> _setStorage2d t s o x y
  [x, y, z]    -> _setStorage3d t s o x y z
  [x, y, z, q] -> _setStorage4d t s o x y z q
  _            -> throwGT4 "setStorage"

_setDim :: IsTensor t => t -> Dim (d::[Nat]) -> HsReal t -> IO ()
_setDim t d v = case dimVals d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> _set1d t x       v
  [x, y]       -> _set2d t x y     v
  [x, y, z]    -> _set3d t x y z   v
  [x, y, z, q] -> _set4d t x y z q v
  _            -> throwGT4 "set"

throwFIXME :: MonadThrow io => String -> String -> io x
throwFIXME fixme msg = throwString $ msg ++ " (FIXME: " ++ fixme ++ ")"

throwNE :: MonadThrow io => String -> io x
throwNE = throwFIXME "make this function only take a non-empty [Nat]"

throwGT4 :: MonadThrow io => String -> io x
throwGT4 fnname = throwFIXME
  ("review how TH supports `" ++ fnname ++ "` operations on > rank-4 tensors")
  (fnname ++ " with >4 rank")

_resizeDim :: IsTensor t => t -> Dim (d::[Nat]) -> IO ()
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
fromList1d :: forall t . (IsTensor t) => [HsReal t] -> IO t
fromList1d l = do
  res :: t <- new' =<< someDimsM [length l]
  mapM_  (upd res) (zip [0..length l - 1] l)
  pure res
 where
  upd :: t -> (Int, HsReal t) -> IO ()
  upd t (idx, v) = someDimsM [idx] >>= \sd -> setDim'_ t sd v

resizeDim :: IsTensor t => t -> Dim (d::[Nat]) -> IO t
resizeDim src d = newClone src >>= \res -> _resizeDim res d >> pure res

resizeDim' :: IsTensor t => t -> SomeDims -> IO t
resizeDim' t (SomeDims d) = resizeDim t d

getDim :: IsTensor t => t -> Dim (d::[Nat]) -> IO (HsReal t)
getDim t d = case dimVals d of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> throwGT4 "get"

getDims :: IsTensor t => t -> IO SomeDims
getDims t = do
  nd <- nDimension t
  ds <- mapM (size t . fromIntegral) [0 .. nd -1]
  someDimsM ds

new :: IsTensor t => Dim (d::[Nat]) -> IO t
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

setDim'_ :: IsTensor t => t -> SomeDims-> HsReal t -> IO ()
setDim'_ t (SomeDims d) v = _setDim t d v

resizeDim'_ :: IsTensor t => t -> SomeDims -> IO ()
resizeDim'_ t (SomeDims d) = _resizeDim t d

getDim' :: IsTensor t => t -> SomeDims -> IO (HsReal t)
getDim' t (SomeDims d) = getDim t d

new' :: IsTensor t => SomeDims -> IO t
new' (SomeDims d) = new d

-- Is this right? why are there three tensors
resizeAs :: forall t . (IsTensor t) => t -> t -> IO t
resizeAs src shape = do
  res <- newClone src
  _resizeAs res shape
  pure res

-- | displaying raw tensor values
printTensor :: forall io t . (IsTensor t, Show (HsReal t)) => t -> IO ()
printTensor t = do
  numDims <- nDimension t
  sizes <- mapM (fmap fromIntegral . size t . fromIntegral) [0..numDims - 1]
  liftIO $ case sizes of
    []  -> putStrLn "Empty Tensor"
    sz@[x] -> do
      putStrLn ""
      putStr "[ "
      mapM_ (get1d t >=> putWithSpace) [ fromIntegral idx | idx <- [0..head sz - 1] ]
      putStrLn "]\n"
    sz@[x,y] -> do
      putStrLn ""
      let pairs = [ (fromIntegral r, fromIntegral c) | r <- [0..sz !! 0 - 1], c <- [0..sz !! 1 - 1] ]
      putStr "[ "
      forM_ pairs $ \(r, c) -> do
        val <- get2d t r c
        if c == fromIntegral (sz !! 1) - 1
        then putStrLn (show val ++ " ]") >> putStr (if fromIntegral r < (sz !! 0) - 1 then "[ " else "")
        else putWithSpace val
    _ -> putStrLn "Can't print this yet."
 where
  putWithSpace :: (Show a) => a -> IO ()
  putWithSpace v = liftIO $ putStr (show v ++ " ")
