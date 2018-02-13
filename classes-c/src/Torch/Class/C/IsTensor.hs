{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Class.C.IsTensor where

import Control.Monad ((>=>), forM_)
import THTypes
import Foreign
import Foreign.C.Types
import Torch.Class.C.Internal
import Torch.Core.Tensor.Dim
import qualified THLongTypes as Long

class IsTensor t where
  clearFlag_ :: t -> Int8 -> IO ()
  tensordata :: t -> IO [HsReal t]
  desc :: t -> IO CTHDescBuff
  expand_ :: t -> t -> Long.Storage -> IO ()
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
  narrow_ :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  new :: IO t
  newClone :: t -> IO t
  newContiguous :: t -> IO t
  newExpand :: t -> Long.Storage -> IO t
  newNarrow :: t -> Int32 -> Int64 -> Int64 -> IO t
  newSelect :: t -> Int32 -> Int64 -> IO t
  newSizeOf :: t -> IO (Long.Storage)
  newStrideOf :: t -> IO (Long.Storage)
  newTranspose :: t -> Int32 -> Int32 -> IO t
  newUnfold :: t -> Int32 -> Int64 -> Int64 -> IO t
  newView :: t -> Long.Storage -> IO t
  newWithSize :: Long.Storage -> Long.Storage -> IO t
  newWithSize1d :: Int64 -> IO t
  newWithSize2d :: Int64 -> Int64 -> IO t
  newWithSize3d :: Int64 -> Int64 -> Int64 -> IO t
  newWithSize4d :: Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage :: HsStorage t -> Int64 -> Long.Storage -> Long.Storage -> IO t
  newWithStorage1d :: HsStorage t -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage2d :: HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage3d :: HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithStorage4d :: HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO t
  newWithTensor :: t -> IO t
  resize_ :: t -> Long.Storage -> Long.Storage -> IO ()
  resize1d_ :: t -> Int64 -> IO ()
  resize2d_ :: t -> Int64 -> Int64 -> IO ()
  resize3d_ :: t -> Int64 -> Int64 -> Int64 -> IO ()
  resize4d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resize5d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  resizeAs_ :: t -> t -> IO ()
  resizeNd_ :: t -> Int32 -> [Int64] -> [Int64] -> IO ()
  retain :: t -> IO ()
  select_ :: t -> t -> Int32 -> Int64 -> IO ()
  set_ :: t -> t -> IO ()
  set1d_ :: t -> Int64 -> HsReal t -> IO ()
  set2d_ :: t -> Int64 -> Int64 -> HsReal t -> IO ()
  set3d_ :: t -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  set4d_ :: t -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal t -> IO ()
  setFlag_ :: t -> Int8 -> IO ()
  setStorage_ :: t -> HsStorage t -> Int64 -> Long.Storage -> Long.Storage -> IO ()
  setStorage1d_ :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage2d_ :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage3d_ :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorage4d_ :: t -> HsStorage t -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO ()
  setStorageNd_ :: t -> HsStorage t -> Int64 -> Int32 -> [Int64] -> [Int64] -> IO ()
  size :: t -> Int32 -> IO Int64
  sizeDesc :: t -> IO CTHDescBuff
  squeeze_ :: t -> t -> IO ()
  squeeze1d_ :: t -> t -> Int32 -> IO ()
  storage :: t -> IO (HsStorage t)
  storageOffset :: t -> IO Int64
  stride :: t -> Int32 -> IO Int64
  transpose_ :: t -> t -> Int32 -> Int32 -> IO ()
  unfold_ :: t -> t -> Int32 -> Int64 -> Int64 -> IO ()
  unsqueeze1d_ :: t -> t -> Int32 -> IO ()

newDim :: IsTensor t => Dim (d::[Nat]) -> IO t
newDim = undefined
setStorageDim_ :: IsTensor t => t -> HsStorage t -> Dim (d::[Nat]) -> IO ()
setStorageDim_ = undefined
setDim_ :: IsTensor t => t -> Dim (d::[Nat]) -> HsReal t -> IO ()
setDim_ = undefined
resizeDim_ :: IsTensor t => t -> Dim (d::[Nat]) -> IO ()
resizeDim_ = undefined
getDim :: IsTensor t => t -> Dim (d::[Nat]) -> IO (HsReal t)
getDim = undefined

newDim' :: IsTensor t => SomeDims -> IO t
newDim' (SomeDims d) = newDim d

setStorageDim'_ :: IsTensor t => t -> HsStorage t -> SomeDims -> IO ()
setStorageDim'_ t s (SomeDims d) = setStorageDim_ t s d

setDim'_ :: IsTensor t => t -> SomeDims-> HsReal t -> IO ()
setDim'_ t (SomeDims d) v = setDim_ t d v

resizeDim'_ :: IsTensor t => t -> SomeDims -> IO ()
resizeDim'_ t (SomeDims d) = resizeDim_ t d

getDim' :: IsTensor t => t -> SomeDims-> IO (HsReal t)
getDim' t (SomeDims d) = getDim t d

-- Is this right?
resizeAs :: IsTensor t => t -> t -> IO t
resizeAs src shape = newClone src >>= \res -> resizeAs_ res shape >> pure res

-- | displaying raw tensor values
printTensor :: forall t . (IsTensor t, Show (HsReal t)) => t -> IO ()
printTensor t = do
  numDims <- nDimension t
  sizes <- mapM (fmap fromIntegral . size t) [0..numDims - 1]
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
