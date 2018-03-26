{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.Tensor where

import Control.Arrow ((***))
import Control.Exception.Safe
import Control.Monad ((>=>), forM_)
import Control.Monad.IO.Class (MonadIO)
import Data.List (genericLength)
import GHC.Int
import Torch.Class.Types
import Torch.Dimensions

class Tensor t where
  clearFlag_ :: t -> Int8 -> io ()
  tensordata :: t -> io [HsReal t]
  expand_ :: t -> t -> SizesStorage t -> io ()
  expandNd_ :: [t] -> [t] -> Int32 -> io ()
  free_ :: t -> io ()
  freeCopyTo_ :: t -> t -> io ()
  get1d :: t -> Int64 -> io (HsReal t)
  get2d :: t -> Int64 -> Int64 -> io (HsReal t)
  get3d :: t -> Int64 -> Int64 -> Int64 -> io (HsReal t)
  get4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> io (HsReal t)
  isContiguous :: t -> io Bool
  isSameSizeAs :: t -> t -> io Bool
  isSetTo :: t -> t -> io Bool
  isSize :: t -> IndexStorage t -> io Bool
  nDimension :: t -> io Int32
  nElement :: t -> io Int64
  narrow_ :: t -> t -> DimVal -> Int64 -> Size -> io ()

  -- | renamed from TH's @new@ because this always returns an empty tensor
  empty :: io t

  newClone :: t -> io t
  newContiguous :: t -> io t
  newExpand :: t -> IndexStorage t -> io t
  newNarrow :: t -> DimVal -> Int64 -> Size -> io t
  newSelect :: t -> DimVal -> Int64 -> io t
  newSizeOf :: t -> io (IndexStorage t)
  newStrideOf :: t -> io (IndexStorage t)
  newTranspose :: t -> DimVal -> DimVal -> io t
  newUnfold :: t -> DimVal -> Int64 -> Int64 -> io t
  newView :: t -> SizesStorage t -> io t
  newWithSize :: SizesStorage t -> StridesStorage t -> io t
  newWithSize1d :: Size -> io t
  newWithSize2d :: Size -> Size -> io t
  newWithSize3d :: Size -> Size -> Size -> io t
  newWithSize4d :: Size -> Size -> Size -> Size -> io t
  newWithStorage :: HsStorage t -> StorageOffset -> SizesStorage t -> StridesStorage t -> io t
  newWithStorage1d :: HsStorage t -> StorageOffset -> (Size, Stride) -> io t
  newWithStorage2d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> io t
  newWithStorage3d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> io t
  newWithStorage4d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> io t
  newWithTensor :: t -> io t
  resize_ :: t -> SizesStorage t -> StridesStorage t -> io ()
  resize1d_ :: t -> Int64 -> io ()
  resize2d_ :: t -> Int64 -> Int64 -> io ()
  resize3d_ :: t -> Int64 -> Int64 -> Int64 -> io ()
  resize4d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> io ()
  resize5d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> io ()
  resizeAs_ :: t -> t -> io ()
  resizeNd_ :: t -> Int32 -> [Size] -> [Stride] -> io ()
  retain :: t -> io ()
  select_ :: t -> t -> DimVal -> Int64 -> io ()
  set_ :: t -> t -> io ()
  set1d_ :: t -> Int64 -> HsReal t -> io ()
  set2d_ :: t -> Int64 -> Int64 -> HsReal t -> io ()
  set3d_ :: t -> Int64 -> Int64 -> Int64 -> HsReal t -> io ()
  set4d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal t -> io ()
  setFlag_ :: t -> Int8 -> io ()
  setStorage_ :: t -> HsStorage t -> StorageOffset -> SizesStorage t -> StridesStorage t -> io ()
  setStorage1d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> io ()
  setStorage2d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> io ()
  setStorage3d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> io ()
  setStorage4d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> io ()
  setStorageNd_ :: t -> HsStorage t -> StorageOffset -> DimVal -> [Size] -> [Stride] -> io ()
  size :: t -> DimVal -> io Size
  sizeDesc :: t -> io (DescBuff t)
  squeeze_ :: t -> t -> io ()
  squeeze1d_ :: t -> t -> DimVal -> io ()
  storage :: t -> io (HsStorage t)
  storageOffset :: t -> io StorageOffset
  stride :: t -> DimVal -> io Stride
  transpose_ :: t -> t -> DimVal -> DimVal -> io ()
  unfold_ :: t -> t -> DimVal -> Size -> Step -> io ()
  unsqueeze1d_ :: t -> t -> DimVal -> io ()

class CPUTensor t where
  desc :: t -> io (DescBuff t)

shape :: Tensor t => t -> IO [Size]
shape t = do
  ds <- nDimension t
  mapM (size t . fromIntegral) [0..ds-1]

inplace :: Tensor t => (t -> IO ()) -> Dim (d::[Nat]) -> IO t
inplace op d = new d >>= \r -> op r >> pure r

inplace' :: Tensor t => (t -> IO ()) -> SomeDims -> IO t
inplace' op (SomeDims d) = inplace op d

inplace1 :: Tensor t => (t -> IO ()) -> t -> IO t
inplace1 op t = getDims t >>= inplace' op

setStorageDim_ :: Tensor t => t -> HsStorage t -> StorageOffset -> [(Size, Stride)] -> IO ()
setStorageDim_ t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> setStorage1d_ t s o x
  [x, y]       -> setStorage2d_ t s o x y
  [x, y, z]    -> setStorage3d_ t s o x y z
  [x, y, z, q] -> setStorage4d_ t s o x y z q
  _            -> throwGT4 "setStorage"

setDim_ :: Tensor t => t -> Dim (d::[Nat]) -> HsReal t -> IO ()
setDim_ t d v = case dimVals d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> set1d_ t x       v
  [x, y]       -> set2d_ t x y     v
  [x, y, z]    -> set3d_ t x y z   v
  [x, y, z, q] -> set4d_ t x y z q v
  _            -> throwGT4 "set"

throwFIXME :: String -> String -> IO x
throwFIXME fixme msg = throwString $ msg ++ " (FIXME: " ++ fixme ++ ")"

throwNE :: String -> IO x
throwNE = throwFIXME "make this function only take a non-empty [Nat]"

throwGT4 :: String -> IO x
throwGT4 fnname = throwFIXME
  ("review how TH supports `" ++ fnname ++ "` operations on > rank-4 tensors")
  (fnname ++ " with >4 rank")

resizeDim_ :: Tensor t => t -> Dim (d::[Nat]) -> IO ()
resizeDim_ t d = case dimVals d of
  []              -> throwNE "can't resize to an empty dimension."
  [x]             -> resize1d_ t x
  [x, y]          -> resize2d_ t x y
  [x, y, z]       -> resize3d_ t x y z
  [x, y, z, q]    -> resize4d_ t x y z q
  [x, y, z, q, w] -> resize5d_ t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
  -- ds              -> resizeNd_ t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")


-- FIXME construct this with TH, not with the setting, which might be doing a second linear pass
fromList1d :: forall t . (Tensor t) => [HsReal t] -> IO t
fromList1d l = do
  res :: t <- new' =<< someDimsM [length l]
  mapM_  (upd res) (zip [0..length l - 1] l)
  pure res
 where
  upd :: t -> (Int, HsReal t) -> IO ()
  upd t (idx, v) = someDimsM [idx] >>= \sd -> setDim'_ t sd v

resizeDim :: Tensor t => t -> Dim (d::[Nat]) -> IO t
resizeDim src d = newClone src >>= \res -> resizeDim_ res d >> pure res

resizeDim' :: Tensor t => t -> SomeDims -> IO t
resizeDim' t (SomeDims d) = resizeDim t d

getDim :: Tensor t => t -> Dim (d::[Nat]) -> IO (HsReal t)
getDim t d = case dimVals d of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> throwGT4 "get"

getDims :: Tensor t => t -> IO SomeDims
getDims t = do
  nd <- nDimension t
  ds <- mapM (size t . fromIntegral) [0 .. nd -1]
  someDimsM ds

new :: Tensor t => Dim (d::[Nat]) -> IO t
new d = case dimVals d of
  []           -> empty
  [x]          -> newWithSize1d x
  [x, y]       -> newWithSize2d x y
  [x, y, z]    -> newWithSize3d x y z
  [x, y, z, q] -> newWithSize4d x y z q
  _ -> do
    t <- empty
    resizeDim_ t d
    pure t

setDim'_ :: Tensor t => t -> SomeDims-> HsReal t -> IO ()
setDim'_ t (SomeDims d) v = setDim_ t d v

resizeDim'_ :: Tensor t => t -> SomeDims -> IO ()
resizeDim'_ t (SomeDims d) = resizeDim_ t d

getDim' :: Tensor t => t -> SomeDims -> IO (HsReal t)
getDim' t (SomeDims d) = getDim t d

new' :: Tensor t => SomeDims -> IO t
new' (SomeDims d) = new d

-- Is this right? why are there three tensors
resizeAs :: Tensor t => t -> t -> IO t
resizeAs src shape = newClone src >>= \res -> resizeAs_ res shape >> pure res

-- | displaying raw tensor values
printTensor :: forall t . (Tensor t, Show (HsReal t)) => t -> IO ()
printTensor t = do
  numDims <- nDimension t
  sizes <- mapM (fmap fromIntegral . size t . fromIntegral) [0..numDims - 1]
  case sizes of
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
  putWithSpace :: Show a => a -> IO ()
  putWithSpace v = putStr (show v ++ " ")
