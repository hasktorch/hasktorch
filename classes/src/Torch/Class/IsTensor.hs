{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.IsTensor where

import Control.Arrow ((***))
import Control.Exception.Safe
import Control.Monad ((>=>), forM_)
import Data.List (genericLength)
import Torch.Types.TH
import GHC.Int
import Torch.Class.Internal
import Torch.Dimensions
import qualified Torch.Types.TH.Long as Long

class IsTensor t where
  clearFlag_ :: t -> Int8 -> IO ()
  tensordata :: t -> IO [HsReal t]
  desc :: t -> IO CTHDescBuff
  expand_ :: t -> t -> SizesStorage -> IO ()
  expandNd_ :: [t] -> [t] -> Int32 -> IO ()
  free_ :: t -> IO ()
  freeCopyTo_ :: t -> t -> IO ()
  get1d :: t -> Int64 -> IO (HsReal t)
  get2d :: t -> Int64 -> Int64 -> IO (HsReal t)
  get3d :: t -> Int64 -> Int64 -> Int64 -> IO (HsReal t)
  get4d :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO (HsReal t)
  isContiguous :: t -> IO Bool
  isSameSizeAs :: t -> t -> IO Bool
  isSetTo :: t -> t -> IO Bool
  isSize :: t -> Long.Storage -> IO Bool
  nDimension :: t -> IO Int32
  nElement :: t -> IO Int64
  narrow_ :: t -> t -> DimVal -> Int64 -> Size -> IO ()
  -- | renamed from TH's @new@ because this always returns an empty tensor
  empty :: IO t
  newClone :: t -> IO t
  newContiguous :: t -> IO t
  newExpand :: t -> Long.Storage -> IO t
  newNarrow :: t -> DimVal -> Int64 -> Size -> IO t
  newSelect :: t -> DimVal -> Int64 -> IO t
  newSizeOf :: t -> IO (Long.Storage)
  newStrideOf :: t -> IO (Long.Storage)
  newTranspose :: t -> DimVal -> DimVal -> IO t
  newUnfold :: t -> DimVal -> Int64 -> Int64 -> IO t
  newView :: t -> SizesStorage -> IO t
  newWithSize :: SizesStorage -> StridesStorage -> IO t
  newWithSize1d :: Size -> IO t
  newWithSize2d :: Size -> Size -> IO t
  newWithSize3d :: Size -> Size -> Size -> IO t
  newWithSize4d :: Size -> Size -> Size -> Size -> IO t
  newWithStorage :: HsStorage t -> StorageOffset -> SizesStorage -> StridesStorage -> IO t
  newWithStorage1d :: HsStorage t -> StorageOffset -> (Size, Stride) -> IO t
  newWithStorage2d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO t
  newWithStorage3d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO t
  newWithStorage4d :: HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO t
  newWithTensor :: t -> IO t
  resize_ :: t -> SizesStorage -> StridesStorage -> IO ()
  resize1d_ :: t -> Int64 -> IO ()
  resize2d_ :: t -> Int64 -> Int64 -> IO ()
  resize3d_ :: t -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resizeAs_ :: t -> t -> IO ()
  resizeNd_ :: t -> Int32 -> [Size] -> [Stride] -> IO ()
  retain :: t -> IO ()
  select_ :: t -> t -> DimVal -> Int64 -> IO ()
  set_ :: t -> t -> IO ()
  set1d_ :: t -> Int64 -> HsReal t -> IO ()
  set2d_ :: t -> Int64 -> Int64 -> HsReal t -> IO ()
  set3d_ :: t -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  set4d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  setFlag_ :: t -> Int8 -> IO ()
  setStorage_ :: t -> HsStorage t -> StorageOffset -> SizesStorage -> StridesStorage -> IO ()
  setStorage1d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> IO ()
  setStorage2d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage3d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage4d_ :: t -> HsStorage t -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorageNd_ :: t -> HsStorage t -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  size :: t -> DimVal -> IO Size
  sizeDesc :: t -> IO CTHDescBuff
  squeeze_ :: t -> t -> IO ()
  squeeze1d_ :: t -> t -> DimVal -> IO ()
  storage :: t -> IO (HsStorage t)
  storageOffset :: t -> IO StorageOffset
  stride :: t -> DimVal -> IO Stride
  transpose_ :: t -> t -> DimVal -> DimVal -> IO ()
  unfold_ :: t -> t -> DimVal -> Size -> Step -> IO ()
  unsqueeze1d_ :: t -> t -> DimVal -> IO ()

shape :: IsTensor t => t -> IO [Size]
shape t = do
  ds <- nDimension t
  mapM (size t . fromIntegral) [0..ds-1]

inplace :: IsTensor t => (t -> IO ()) -> Dim (d::[Nat]) -> IO t
inplace op d = new d >>= \r -> op r >> pure r

inplace' :: IsTensor t => (t -> IO ()) -> SomeDims -> IO t
inplace' op (SomeDims d) = inplace op d

inplace1 :: IsTensor t => (t -> IO ()) -> t -> IO t
inplace1 op t = getDims t >>= inplace' op

setStorageDim_ :: IsTensor t => t -> HsStorage t -> StorageOffset -> [(Size, Stride)] -> IO ()
setStorageDim_ t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> setStorage1d_ t s o x
  [x, y]       -> setStorage2d_ t s o x y
  [x, y, z]    -> setStorage3d_ t s o x y z
  [x, y, z, q] -> setStorage4d_ t s o x y z q
  _            -> throwGT4 "setStorage"

setDim_ :: IsTensor t => t -> Dim (d::[Nat]) -> HsReal t -> IO ()
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

resizeDim_ :: IsTensor t => t -> Dim (d::[Nat]) -> IO ()
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
fromList1d :: forall t . (IsTensor t) => [HsReal t] -> IO t
fromList1d l = do
  res :: t <- new' =<< someDimsM [length l]
  mapM_  (upd res) (zip [0..length l - 1] l)
  pure res
 where
  upd :: t -> (Int, HsReal t) -> IO ()
  upd t (idx, v) = someDimsM [idx] >>= \sd -> setDim'_ t sd v

resizeDim :: IsTensor t => t -> Dim (d::[Nat]) -> IO t
resizeDim src d = newClone src >>= \res -> resizeDim_ res d >> pure res

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
    resizeDim_ t d
    pure t

setDim'_ :: IsTensor t => t -> SomeDims-> HsReal t -> IO ()
setDim'_ t (SomeDims d) v = setDim_ t d v

resizeDim'_ :: IsTensor t => t -> SomeDims -> IO ()
resizeDim'_ t (SomeDims d) = resizeDim_ t d

getDim' :: IsTensor t => t -> SomeDims -> IO (HsReal t)
getDim' t (SomeDims d) = getDim t d

new' :: IsTensor t => SomeDims -> IO t
new' (SomeDims d) = new d

-- Is this right? why are there three tensors
resizeAs :: IsTensor t => t -> t -> IO t
resizeAs src shape = newClone src >>= \res -> resizeAs_ res shape >> pure res

-- | displaying raw tensor values
printTensor :: forall t . (IsTensor t, Show (HsReal t)) => t -> IO ()
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
