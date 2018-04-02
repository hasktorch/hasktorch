{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Class.Tensor.Static where

import Data.Proxy
import GHC.TypeLits
import GHC.Int
import GHC.Natural
import Control.Exception.Safe
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.Prelude.List
import Data.Singletons.Prelude.Num
import Control.Monad

import Torch.Dimensions

import Torch.Class.Types
-- import Torch.Class.Tensor as X hiding (new, fromList1d, resizeDim)
import qualified Torch.Class.Tensor as Dynamic
import qualified Torch.Types.TH as TH (IndexStorage)

class Tensor t where
  clearFlag_ :: Dimensions d => t d -> Int8 -> IO ()
  tensordata :: Dimensions d => t d -> IO [HsReal (t d)]
  free_ :: Dimensions d => t d -> IO ()
  freeCopyTo_ :: (Dimensions d, Dimensions d') => t d -> t d' -> IO ()
  get1d :: t d -> Int64 -> IO (HsReal (t d))
  get2d :: t d -> Int64 -> Int64 -> IO (HsReal (t d))
  get3d :: t d -> Int64 -> Int64 -> Int64 -> IO (HsReal (t d))
  get4d :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> IO (HsReal (t d))
  isContiguous :: t d -> IO Bool
  isSetTo :: t d -> t d' -> IO Bool
  isSize :: t d -> TH.IndexStorage -> IO Bool
  nDimension :: t d -> IO Int32
  nElement :: t d -> IO Int64
  narrow_ :: t d -> t d' -> DimVal -> Int64 -> Size -> IO ()

  -- | renamed from TH's @new@ because this always returns an empty tensor
  empty :: IO (t d)

  newClone :: (t d) -> IO (t d)
  newContiguous :: t d -> IO (t d')
  newNarrow :: t d -> DimVal -> Int64 -> Size -> IO (t d')
  newSelect :: t d -> DimVal -> Int64 -> IO (t d')
  newSizeOf :: t d -> IO TH.IndexStorage
  newStrideOf :: t d -> IO TH.IndexStorage
  newTranspose :: t d -> DimVal -> DimVal -> IO (t d')
  newUnfold :: t d -> DimVal -> Int64 -> Int64 -> IO (t d')
  newView :: t d -> SizesStorage (t d) -> IO (t d')
  newWithSize :: SizesStorage (t d) -> StridesStorage (t d) -> IO (t d)
  newWithSize1d :: Size -> IO (t d)
  newWithSize2d :: Size -> Size -> IO (t d)
  newWithSize3d :: Size -> Size -> Size -> IO (t d)
  newWithSize4d :: Size -> Size -> Size -> Size -> IO (t d)
  newWithStorage :: HsStorage (t d) -> StorageOffset -> SizesStorage (t d) -> StridesStorage (t d) -> IO (t d)
  newWithStorage1d :: HsStorage (t d) -> StorageOffset -> (Size, Stride) -> IO (t d)
  newWithStorage2d :: HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO (t d)
  newWithStorage3d :: HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO (t d)
  newWithStorage4d :: HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO (t d)
  newWithTensor :: t d -> IO (t d)
  resize_ :: t d -> SizesStorage (t d) -> StridesStorage (t d) -> IO (t d')
  resize1d_ :: t d -> Int64 -> IO (t d')
  resize2d_ :: t d -> Int64 -> Int64 -> IO (t d')
  resize3d_ :: t d -> Int64 -> Int64 -> Int64 -> IO (t d')
  resize4d_ :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> IO (t d')
  resize5d_ :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO (t d')
  resizeAs_ :: t d -> t d' -> IO (t d')
  resizeNd_ :: t d -> Int32 -> [Size] -> [Stride] -> IO (t d')
  retain :: t d -> IO ()
  select_ :: t d -> t d' -> DimVal -> Int64 -> IO ()
  set_ :: t d -> t d -> IO ()
  set1d_ :: t d -> Int64 -> HsReal (t d) -> IO ()
  set2d_ :: t d -> Int64 -> Int64 -> HsReal (t d) -> IO ()
  set3d_ :: t d -> Int64 -> Int64 -> Int64 -> HsReal (t d) -> IO ()
  set4d_ :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal (t d) -> IO ()
  setFlag_ :: t d -> Int8 -> IO ()
  setStorage_   :: t d -> HsStorage (t d) -> StorageOffset -> SizesStorage (t d) -> StridesStorage (t d) -> IO ()
  setStorage1d_ :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> IO ()
  setStorage2d_ :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage3d_ :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorage4d_ :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  setStorageNd_ :: t d -> HsStorage (t d) -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  size :: t d -> DimVal -> IO Size
  sizeDesc :: t d -> IO (DescBuff (t d))
  squeeze_ :: t d -> t d' -> IO ()
  squeeze1d_ :: t d -> t d' -> DimVal -> IO ()
  storage :: t d -> IO (HsStorage (t d))
  storageOffset :: t d -> IO StorageOffset
  stride :: t d -> DimVal -> IO Stride
  transpose_ :: t d -> t d' -> DimVal -> DimVal -> IO ()
  unfold_ :: t d -> t d' -> DimVal -> Size -> Step -> IO ()
  unsqueeze1d_ :: t d -> t d' -> DimVal -> IO ()

  -- New for static tensors
  fromList1d :: [HsReal (t '[n])] -> IO (t '[n])

  -- Modified for static tensors
  isSameSizeAs :: (Dimensions d, Dimensions d') => t d -> t d' -> Bool

type Static t d =
  ( Tensor t
  , IsStatic (t d)
  , Num (HsReal (IndexDynamic (AsDynamic (t d))))
  , Dynamic.Tensor (AsDynamic (t d))
  )
type Static2 t d d' = 
  ( Static t d
  , Static t d'
  , AsDynamic (t d) ~ AsDynamic (t d')
  , IndexDynamic (AsDynamic (t d)) ~ IndexDynamic (AsDynamic (t d'))
  )

shape :: Tensor t => t d -> IO [Size]
shape t = do
  ds <- nDimension t
  mapM (size t . fromIntegral) [0..ds-1]

withInplace :: forall t d . (Dimensions d, Tensor t) => (t d -> IO ()) -> IO (t d)
withInplace op = new >>= \r -> op r >> pure r

throwFIXME :: MonadThrow io => String -> String -> io x
throwFIXME fixme msg = throwString $ msg ++ " (FIXME: " ++ fixme ++ ")"

throwNE :: MonadThrow io => String -> io x
throwNE = throwFIXME "make this function only take a non-empty [Nat]"

throwGT4 :: MonadThrow io => String -> io x
throwGT4 fnname = throwFIXME
  ("review how TH supports `" ++ fnname ++ "` operations on > rank-4 tensors")
  (fnname ++ " with >4 rank")


setStorageDim_ :: Tensor t => t d -> HsStorage (t d) -> StorageOffset -> [(Size, Stride)] -> IO ()
setStorageDim_ t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> setStorage1d_ t s o x
  [x, y]       -> setStorage2d_ t s o x y
  [x, y, z]    -> setStorage3d_ t s o x y z
  [x, y, z, q] -> setStorage4d_ t s o x y z q
  _            -> throwGT4 "setStorage"

setDim_ :: forall t d d' . (Dimensions d', Tensor t) => t d -> Dim d' -> HsReal (t d) -> IO ()
setDim_ t d v = case dimVals d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> set1d_ t x       v
  [x, y]       -> set2d_ t x y     v
  [x, y, z]    -> set3d_ t x y z   v
  [x, y, z, q] -> set4d_ t x y z q v
  _            -> throwGT4 "set"

-- setDim'_ :: (Dimensions d, Tensor t) => t d -> SomeDims -> HsReal (t d) -> IO ()
-- setDim'_ t (SomeDims d) v = setDim_ t d v

getDim :: (Dimensions d', Tensor t) => t d -> Dim (d'::[Nat]) -> IO (HsReal (t d))
getDim t d = case dimVals d of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> throwGT4 "get"

getDims :: Tensor t => t d -> IO SomeDims
getDims t = do
  nd <- nDimension t
  ds <- mapM (size t . fromIntegral) [0 .. nd -1]
  someDimsM ds

new :: forall t d . (Dimensions d, Tensor t) => IO (t d)
new = case dimVals d of
  []           -> empty
  [x]          -> newWithSize1d x
  [x, y]       -> newWithSize2d x y
  [x, y, z]    -> newWithSize3d x y z
  [x, y, z, q] -> newWithSize4d x y z q
  _ -> empty >>= resizeDim_
 where
  d :: Dim d
  d = dim

-- NOTE: This is copied from the dynamic version to keep the constraints clean and is _unsafe_
resizeDim_ :: forall t d d' . (Tensor t, Dimensions d') => t d -> IO (t d')
resizeDim_ t = case dimVals d of
  []              -> throwNE "can't resize to an empty dimension."
  [x]             -> resize1d_ t x
  [x, y]          -> resize2d_ t x y
  [x, y, z]       -> resize3d_ t x y z
  [x, y, z, q]    -> resize4d_ t x y z q
  [x, y, z, q, w] -> resize5d_ t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
 where
  d :: Dim d'
  d = dim
  -- ds              -> resizeNd_ t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")

resizeAs :: forall t d d' . (Dimensions d, Dimensions d', Tensor t) => t d -> IO (t d')
resizeAs src = do
  res <- newClone src
  shape <- new
  resizeAs_ res shape

newIx :: forall t d d'
  . (Dimensions d')
  => Dynamic.Tensor (AsDynamic (IndexTensor (t d) d'))
  => IsStatic (IndexTensor (t d) d')
  => IO (IndexTensor (t d) d')
newIx = asStatic <$> Dynamic.new (dim :: Dim d')


-- FIXME construct this with TH, not with the setting, which might be doing a second linear pass
fromListIx
  :: forall t d n . (KnownNatDim n, Dimensions '[n], IsStatic (IndexTensor (t d) '[n]))
  => Num (HsReal (AsDynamic (IndexTensor (t d) '[n])))
  => Dynamic.Tensor (AsDynamic (IndexTensor (t d) '[n]))
  => Dimensions d
  => Proxy (t d) -> Dim '[n] -> [HsReal (AsDynamic (IndexTensor (t d) '[n]))] -> IO (IndexTensor (t d) '[n])
fromListIx _ _ l = asStatic <$> (Dynamic.fromList1d l)


-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME: There might be a faster way to do this with newWithSize
fromList
  :: forall t d
  .  (KnownNatDim (Product d), Dimensions d, Tensor t)
  => [HsReal (t '[Product d])] -> IO (t d)
fromList l = do
  oneD :: t '[Product d] <- fromList1d l
  resizeDim_ oneD

newTranspose2d
  :: forall t r c . (KnownNat2 r c, Tensor t, Dimensions '[r, c], Dimensions '[c, r])
  => t '[r, c] -> IO (t '[c, r])
newTranspose2d t = newTranspose t 1 0

-- -- | Expand a vector by copying into a matrix by set dimensions
-- -- TODO - generalize this beyond the matrix case
-- expand2d
--   :: forall t d1 d2 . (KnownNatDim2 d1 d2)
--   => StaticConstraint2 t '[d2, d1] '[d1]
--   => Dynamic.TensorMath (AsDynamic (t '[d1])) -- for 'Dynamic.constant' which uses 'Torch.Class.Tensor.Math.fill'
--   => t '[d1] -> IO (t '[d2, d1])
-- expand2d t = do
--   res :: AsDynamic (t '[d2, d1]) <- Dynamic.constant (dim :: Dim '[d2, d1]) 0
--   s :: LongStorage <- Storage.newWithSize2 s2 s1
--   Dynamic.expand_ res (asDynamic t) s
--   pure (asStatic res)
--   where
--     s1, s2 :: Integer
--     s1 = natVal (Proxy :: Proxy d1)
--     s2 = natVal (Proxy :: Proxy d2)

getElem2d
  :: forall t n m . (KnownNatDim2 n m)
  => Tensor t => Dimensions '[n, m]
  => t '[n, m] -> Natural -> Natural -> IO (HsReal (t '[n, m]))
getElem2d t r c
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = get2d t (fromIntegral r) (fromIntegral c)

setElem2d
  :: forall t n m ns . (KnownNatDim2 n m)
  => Tensor t => Dimensions '[n, m]
  => Dimensions ns
  => t '[n, m] -> Natural -> Natural -> HsReal (t '[n, m]) -> IO ()
setElem2d t r c v
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = set2d_ t (fromIntegral r) (fromIntegral c) v


-- | displaying raw tensor values
printTensor :: forall t d . (Dimensions d, Tensor t, Show (HsReal (t d))) => t d -> IO ()
printTensor t = do
  case dimVals (dim :: Dim d) of
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
  putWithSpace v = putStr (show v ++ " ")
