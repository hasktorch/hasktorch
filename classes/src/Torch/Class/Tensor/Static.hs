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
import Data.List.NonEmpty (NonEmpty)

import Torch.Class.Types
-- import Torch.Class.Tensor as X hiding (new, fromList1d, resizeDim)
import qualified Torch.Class.Tensor as Dynamic
import qualified Torch.Types.TH as TH (IndexStorage)

class IsTensor t where
  _clearFlag :: Dimensions d => t d -> Int8 -> IO ()
  tensordata :: Dimensions d => t d -> IO [HsReal (t d)]
  _free :: Dimensions d => t d -> IO ()
  _freeCopyTo :: Dimensions2 d d' => t d -> t d' -> IO ()
  get1d :: t d -> Int64 -> IO (HsReal (t d))
  get2d :: t d -> Int64 -> Int64 -> IO (HsReal (t d))
  get3d :: t d -> Int64 -> Int64 -> Int64 -> IO (HsReal (t d))
  get4d :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> IO (HsReal (t d))
  isContiguous :: t d -> IO Bool
  isSetTo :: t d -> t d' -> IO Bool
  isSize :: t d -> TH.IndexStorage -> IO Bool
  nDimension :: t d -> IO Int32
  nElement :: t d -> IO Int64
  _narrow :: t d -> t d' -> DimVal -> Int64 -> Size -> IO ()

  -- | renamed from TH's @new@ because this always returns an empty tensor
  -- FIXME: this __technically should be @IO (t '[])@, but if you leave it as-is
  -- the types line-up nicely (and we currently don't use rank-0 tensors).
  empty :: IO (t d)
  newExpand :: Dimensions2 d d' => t d -> TH.IndexStorage -> IO (t d')
  expand    :: Dimensions2 d d' => t d' -> t d -> TH.IndexStorage -> IO ()
  expandNd  :: Dimensions d => NonEmpty (t d) -> NonEmpty (t d) -> Int -> IO ()
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
  _resize :: t d -> SizesStorage (t d) -> StridesStorage (t d) -> IO (t d')
  _resize1d :: t d -> Int64 -> IO (t d')
  _resize2d :: t d -> Int64 -> Int64 -> IO (t d')
  _resize3d :: t d -> Int64 -> Int64 -> Int64 -> IO (t d')
  _resize4d :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> IO (t d')
  _resize5d :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> Int64 -> IO (t d')
  _resizeAs :: t d -> t d' -> IO (t d')
  _resizeNd :: t d -> Int32 -> [Size] -> [Stride] -> IO (t d')
  retain :: t d -> IO ()
  _select :: t d -> t d' -> DimVal -> Int64 -> IO ()
  _set :: t d -> t d -> IO ()
  _set1d :: t d -> Int64 -> HsReal (t d) -> IO ()
  _set2d :: t d -> Int64 -> Int64 -> HsReal (t d) -> IO ()
  _set3d :: t d -> Int64 -> Int64 -> Int64 -> HsReal (t d) -> IO ()
  _set4d :: t d -> Int64 -> Int64 -> Int64 -> Int64 -> HsReal (t d) -> IO ()
  _setFlag :: t d -> Int8 -> IO ()
  _setStorage   :: t d -> HsStorage (t d) -> StorageOffset -> SizesStorage (t d) -> StridesStorage (t d) -> IO ()
  _setStorage1d :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> IO ()
  _setStorage2d :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> IO ()
  _setStorage3d :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  _setStorage4d :: t d -> HsStorage (t d) -> StorageOffset -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> (Size, Stride) -> IO ()
  _setStorageNd :: t d -> HsStorage (t d) -> StorageOffset -> DimVal -> [Size] -> [Stride] -> IO ()
  size :: t d -> DimVal -> IO Size
  sizeDesc :: t d -> IO (DescBuff (t d))
  _squeeze :: t d -> t d' -> IO ()
  _squeeze1d :: t d -> t d' -> DimVal -> IO ()
  storage :: t d -> IO (HsStorage (t d))
  storageOffset :: t d -> IO StorageOffset
  stride :: t d -> DimVal -> IO Stride
  _transpose :: t d -> t d' -> DimVal -> DimVal -> IO ()
  _unfold :: t d -> t d' -> DimVal -> Size -> Step -> IO ()
  _unsqueeze1d :: t d -> t d' -> DimVal -> IO ()

  -- New for static tensors
  fromList1d :: [HsReal (t '[n])] -> IO (t '[n])

  -- Modified for static tensors
  isSameSizeAs :: (Dimensions d, Dimensions d') => t d -> t d' -> Bool

type Static t d =
  ( IsTensor t
  , IsStatic (t d)
  , Num (HsReal (IndexDynamic (AsDynamic (t d))))
  , Dynamic.IsTensor (AsDynamic (t d))
  )
type Static2 t d d' = 
  ( Static t d
  , Static t d'
  , AsDynamic (t d) ~ AsDynamic (t d')
  , IndexDynamic (AsDynamic (t d)) ~ IndexDynamic (AsDynamic (t d'))
  )

shape :: IsTensor t => t d -> IO [Size]
shape t = do
  ds <- nDimension t
  mapM (size t . fromIntegral) [0..ds-1]

withNew :: forall t d . (Dimensions d, IsTensor t) => (t d -> IO ()) -> IO (t d)
withNew op = new >>= \r -> op r >> pure r

-- I think we can get away with this for most creations. Torch does the resizing in C.
withEmpty :: Dimensions d => IsTensor t => (t d -> IO ()) -> IO (t d)
withEmpty op = empty >>= \r -> op r >> pure r

type CoerceDims t d d' = (Dimensions2 d d', IsStatic (t d), AsDynamic (t d) ~ AsDynamic (t d'), IsStatic (t d'))

useSudo :: (CoerceDims t d d') => t d -> t d'
useSudo = asStatic . asDynamic

-- This is actually 'inplace'. Dimensions may change from original tensor given Torch resizing.
withInplace :: (IsTensor t, Dimensions d) => t d -> (t d -> t d -> IO ()) -> IO (t d)
withInplace t op = op t t >> pure t

-- This is actually 'inplace'. Dimensions may change from original tensor given Torch resizing.
sudoInplace
  :: forall t d d' . (IsTensor t, CoerceDims t d d')
  => t d -> (t d' -> t d -> IO ()) -> IO (t d')
sudoInplace t op = op ret t >> pure ret
  where
    ret :: t d'
    ret = useSudo t

throwFIXME :: MonadThrow io => String -> String -> io x
throwFIXME fixme msg = throwString $ msg ++ " (FIXME: " ++ fixme ++ ")"

throwNE :: MonadThrow io => String -> io x
throwNE = throwFIXME "make this function only take a non-empty [Nat]"

throwGT4 :: MonadThrow io => String -> io x
throwGT4 fnname = throwFIXME
  ("review how TH supports `" ++ fnname ++ "` operations on > rank-4 tensors")
  (fnname ++ " with >4 rank")


_setStorageDim :: IsTensor t => t d -> HsStorage (t d) -> StorageOffset -> [(Size, Stride)] -> IO ()
_setStorageDim t s o = \case
  []           -> throwNE "can't setStorage on an empty dimension."
  [x]          -> _setStorage1d t s o x
  [x, y]       -> _setStorage2d t s o x y
  [x, y, z]    -> _setStorage3d t s o x y z
  [x, y, z, q] -> _setStorage4d t s o x y z q
  _            -> throwGT4 "setStorage"

_setDim :: forall t d d' . (Dimensions d', IsTensor t) => t d -> Dim d' -> HsReal (t d) -> IO ()
_setDim t d v = case dimVals d of
  []           -> throwNE "can't set on an empty dimension."
  [x]          -> _set1d t x       v
  [x, y]       -> _set2d t x y     v
  [x, y, z]    -> _set3d t x y z   v
  [x, y, z, q] -> _set4d t x y z q v
  _            -> throwGT4 "set"

-- setDim'_ :: (Dimensions d, IsTensor t) => t d -> SomeDims -> HsReal (t d) -> IO ()
-- setDim'_ t (SomeDims d) v = _setDim t d v

getDim :: (Dimensions d', IsTensor t) => t d -> Dim (d'::[Nat]) -> IO (HsReal (t d))
getDim t d = case dimVals d of
  []           -> throwNE "can't lookup an empty dimension"
  [x]          -> get1d t x
  [x, y]       -> get2d t x y
  [x, y, z]    -> get3d t x y z
  [x, y, z, q] -> get4d t x y z q
  _            -> throwGT4 "get"

getDims :: IsTensor t => t d -> IO SomeDims
getDims t = do
  nd <- nDimension t
  ds <- mapM (size t . fromIntegral) [0 .. nd -1]
  someDimsM ds

new :: forall t d . (Dimensions d, IsTensor t) => IO (t d)
new = case dimVals d of
  []           -> empty
  [x]          -> newWithSize1d x
  [x, y]       -> newWithSize2d x y
  [x, y, z]    -> newWithSize3d x y z
  [x, y, z, q] -> newWithSize4d x y z q
  _ -> empty >>= _resizeDim
 where
  d :: Dim d
  d = dim

-- NOTE: This is copied from the dynamic version to keep the constraints clean and is _unsafe_
_resizeDim :: forall t d d' . (IsTensor t, Dimensions d') => t d -> IO (t d')
_resizeDim t = case dimVals d of
  []              -> throwNE "can't resize to an empty dimension."
  [x]             -> _resize1d t x
  [x, y]          -> _resize2d t x y
  [x, y, z]       -> _resize3d t x y z
  [x, y, z, q]    -> _resize4d t x y z q
  [x, y, z, q, w] -> _resize5d t x y z q w
  _ -> throwFIXME "this should be doable with resizeNd" "resizeDim"
 where
  d :: Dim d'
  d = dim
  -- ds              -> _resizeNd t (genericLength ds) ds
                            -- (error "resizeNd_'s stride should be given a c-NULL or a haskell-nullPtr")

resizeAs :: forall t d d' . (Dimensions d, Dimensions d', IsTensor t) => t d -> IO (t d')
resizeAs src = do
  res <- newClone src
  shape <- new
  _resizeAs res shape

newIx :: forall t d d'
  . (Dimensions d')
  => Dynamic.IsTensor (AsDynamic (IndexTensor (t d) d'))
  => IsStatic (IndexTensor (t d) d')
  => IO (IndexTensor (t d) d')
newIx = asStatic <$> Dynamic.new (dim :: Dim d')


-- FIXME construct this with TH, not with the setting, which might be doing a second linear pass
fromListIx
  :: forall t d n . (KnownNatDim n, Dimensions '[n], IsStatic (IndexTensor (t d) '[n]))
  => Num (HsReal (AsDynamic (IndexTensor (t d) '[n])))
  => Dynamic.IsTensor (AsDynamic (IndexTensor (t d) '[n]))
  => Dimensions d
  => Proxy (t d) -> Dim '[n] -> [HsReal (AsDynamic (IndexTensor (t d) '[n]))] -> IO (IndexTensor (t d) '[n])
fromListIx _ _ l = asStatic <$> (Dynamic.fromList1d l)


-- | Initialize a tensor of arbitrary dimension from a list
-- FIXME: There might be a faster way to do this with newWithSize
fromList
  :: forall t d
  .  (KnownNatDim (Product d), Dimensions d, IsTensor t)
  => [HsReal (t '[Product d])] -> IO (t d)
fromList l = do
  oneD :: t '[Product d] <- fromList1d l
  _resizeDim oneD

newTranspose2d
  :: forall t r c . (KnownNat2 r c, IsTensor t, Dimensions '[r, c], Dimensions '[c, r])
  => t '[r, c] -> IO (t '[c, r])
newTranspose2d t = newTranspose t 1 0

-- -- | Expand a vector by copying into a matrix by set dimensions
-- -- TODO - generalize this beyond the matrix case
-- expand2d
--   :: forall t d1 d2 . (KnownNatDim2 d1 d2)
--   => StaticConstraint2 t '[d2, d1] '[d1]
--   => Dynamic.IsTensorMath (AsDynamic (t '[d1])) -- for 'Dynamic.constant' which uses 'Torch.Class.IsTensor.Math.fill'
--   => t '[d1] -> IO (t '[d2, d1])
-- expand2d t = do
--   res :: AsDynamic (t '[d2, d1]) <- Dynamic.constant (dim :: Dim '[d2, d1]) 0
--   s :: LongStorage <- Storage.newWithSize2 s2 s1
--   Dynamic._expand res (asDynamic t) s
--   pure (asStatic res)
--   where
--     s1, s2 :: Integer
--     s1 = natVal (Proxy :: Proxy d1)
--     s2 = natVal (Proxy :: Proxy d2)

getElem2d
  :: forall t n m . (KnownNatDim2 n m)
  => IsTensor t => Dimensions '[n, m]
  => t '[n, m] -> Natural -> Natural -> IO (HsReal (t '[n, m]))
getElem2d t r c
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = get2d t (fromIntegral r) (fromIntegral c)

setElem2d
  :: forall t n m ns . (KnownNatDim2 n m)
  => IsTensor t => Dimensions '[n, m]
  => Dimensions ns
  => t '[n, m] -> Natural -> Natural -> HsReal (t '[n, m]) -> IO ()
setElem2d t r c v
  | r > fromIntegral (natVal (Proxy :: Proxy n)) ||
    c > fromIntegral (natVal (Proxy :: Proxy m))
      = throwString "Indices out of bounds"
  | otherwise = _set2d t (fromIntegral r) (fromIntegral c) v


-- | displaying raw tensor values
printTensor :: forall t d . (Dimensions d, IsTensor t, Show (HsReal (t d))) => t d -> IO ()
printTensor t = do
  case dimVals (dim :: Dim d) of
    []  -> putStrLn "Empty IsTensor"
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
