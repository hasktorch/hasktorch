{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Core.Tensor.Generic
  ( flatten
  , randInit
  , randInit'
  , constant
  , constant'
  , applyInPlaceFn
  , dimList
  , dimView
  , getDynamicDim
  , fillZeros
  , inplaceFill
  , genericNew
  , genericNew'
  , genericGet
  , genericGet'


  , GenericOps(..)
  , GenericMath(..)
  , GenericRandom(..)

  , dispRaw
  ) where

import Numeric.Dimensions (Dim(..), someDimsVal)
import Foreign (Ptr)
import Foreign.C.Types
import Data.Maybe (fromJust)
import qualified Numeric as Num (showGFloat)

import Torch.Core.Internal (impossible, i2cll)
import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Generic.Internal
import Torch.Core.Tensor.Generic.Math
import Torch.Core.Tensor.Generic.Random
import Torch.Core.Tensor.Generic.Ops

import THTypes

-- | flatten a CTHDoubleTensor into a list
flatten :: GenericOps t => Ptr t -> [HaskReal t]
flatten tensor =
  case map getDim [0 .. nDimension tensor - 1] of
    []           -> mempty
    [x]          -> get1d tensor <$> range x
    [x, y]       -> get2d tensor <$> range x <*> range y
    [x, y, z]    -> get3d tensor <$> range x <*> range y <*> range z
    [x, y, z, q] -> get4d tensor <$> range x <*> range y <*> range z <*> range q
    _ -> error "TH doesn't support getting tensors higher than 4-dimensions"
  where
    getDim :: CInt -> Int
    getDim = fromIntegral . size tensor

    range :: Integral i => Int -> [i]
    range mx = [0 .. fromIntegral mx - 1]

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInit
  :: (GenericMath t, GenericRandom t, GenericOps t, Num (HaskReal t))
  => Ptr CTHGenerator
  -> Dim (dims :: [k])
  -> CDouble
  -> CDouble
  -> IO (Ptr t)
randInit gen dims lower upper = do
  t <- constant dims 0
  uniform t gen lower upper
  pure t

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInit'
  :: (GenericMath t, GenericRandom t, GenericOps t, Num (HaskReal t))
  => Ptr CTHGenerator
  -> SomeDims
  -> CDouble
  -> CDouble
  -> IO (Ptr t)
randInit' gen dims lower upper = do
  t <- constant' dims 0
  uniform t gen lower upper
  pure t

-- | Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
-- fillDouble :: (GenericMath t, GenericOps t) => HaskReal t -> Ptr t -> IO ()
-- fillDouble = flip fill . realToFrac

-- | Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
-- tensorRaw :: Dim (ns::[k]) -> Double -> IO TensorDoubleRaw
constant :: forall ns t . (GenericMath t, GenericOps t) => Dim (ns::[k]) -> HaskReal t -> IO (Ptr t)
constant dims value = do
  newPtr <- genericNew dims
  fill newPtr value
  pure newPtr

-- | Create a new (double) tensor of specified dimensions and fill it with 0
-- safe version
constant' :: forall ns t . (GenericMath t, GenericOps t) => SomeDims -> HaskReal t -> IO (Ptr t)
constant' dims value = do
  newPtr <- genericNew' dims
  fill newPtr value
  pure newPtr

genericNew :: GenericOps t => Dim (ns::[k]) -> IO (Ptr t)
genericNew = onDims fromIntegral
  new
  newWithSize1d
  newWithSize2d
  newWithSize3d
  newWithSize4d

genericNew' :: GenericOps t => SomeDims -> IO (Ptr t)
genericNew' = onDims' fromIntegral
  new
  newWithSize1d
  newWithSize2d
  newWithSize3d
  newWithSize4d

genericGet :: GenericOps t => Ptr t -> Dim (ns::[k]) -> HaskReal t
genericGet t = onDims fromIntegral
  (impossible "0-rank will never be called")
  (get1d t)
  (get2d t)
  (get3d t)
  (get4d t)

genericGet' :: GenericOps t => Ptr t -> SomeDims -> HaskReal t
genericGet' t = onDims' fromIntegral
  (impossible "0-rank will never be called")
  (get1d t)
  (get2d t)
  (get3d t)
  (get4d t)


-- |apply a tensor transforming function to a tensor
applyInPlaceFn :: GenericOps t => (Ptr t -> Ptr t -> IO ()) -> Ptr t -> IO (Ptr t)
applyInPlaceFn f t1 = do
  r_ <- new
  f r_ t1
  pure r_

-- | Dimensions of a raw tensor as a list
dimList :: GenericOps t => Ptr t -> [Int]
dimList t = getDim <$> [0 .. nDimension t - 1]
  where
    getDim :: CInt -> Int
    getDim = fromIntegral . size t

-- |Dimensions of a raw tensor as a TensorDim value
dimView :: GenericOps t => Ptr t -> DimView
dimView t =
  case length sz of
    0 -> D0
    1 -> D1 (at 0)
    2 -> D2 (at 0) (at 1)
    3 -> D3 (at 0) (at 1) (at 2)
    4 -> D4 (at 0) (at 1) (at 2) (at 3)
    5 -> D5 (at 0) (at 1) (at 2) (at 3) (at 5)
    _ -> undefined -- TODO - make this safe
  where
    sz :: [Int]
    sz = dimList t

    at :: Int -> Int
    at n = fromIntegral (sz !! n)

-- | Dimensions of a raw tensor as a SomeDims value.
getDynamicDim :: GenericOps t => Ptr t -> SomeDims
getDynamicDim
-- Note: we can safely call 'fromJust' since these values are maintained by TH which does the bounds-checking
  = fromJust . someDimsVal . dimList

-- | Fill a raw Double tensor with 0.0
fillZeros :: (GenericMath t, GenericOps t, Num (HaskReal t)) => Ptr t -> IO (Ptr t)
fillZeros t = fill t 0 >> pure t

-- ========================================================================= --
-- TO BE REMOVED: dispRaw
-- ========================================================================= --

-- | displaying raw tensor values
dispRaw :: forall t . (GenericOps t, Show (HaskReal t)) => Ptr t -> IO ()
dispRaw tensor
  | length sz == 0 = putStrLn "Empty Tensor"
  | length sz == 1 = do
      putStrLn ""
      let indexes = [ fromIntegral idx :: CLLong
                    | idx <- [0..head sz - 1]
                    ]
      putStr "[ "
      mapM_ (\idx -> putStr $ showLim (get1d tensor idx) ++ " ")
        indexes
      putStrLn "]\n"
  | length sz == 2 = do
      putStrLn ""
      let
        pairs :: [(CLLong, CLLong)]
        pairs = [ (fromIntegral r, fromIntegral c)
                | r <- [0..sz !! 0 - 1]
                , c <- [0..sz !! 1 - 1]
                ]
      putStr ("[ " :: String)
      mapM_ (\(r, c) -> do
                let val = get2d tensor r c
                if c == fromIntegral (sz !! 1) - 1
                  then do
                  putStrLn (showLim val ++ " ]")
                  putStr (if (fromIntegral r :: Int) < (sz !! 0) - 1
                          then "[ " :: String
                          else "")
                  else
                  putStr $ showLim val ++ " "
            ) pairs
  | otherwise = putStrLn "Can't print this yet."
  where
    sizes :: Ptr t -> [Int]
    sizes t = fmap (fromIntegral . size t) [0..nDimension t - 1]

    showLim :: HaskReal t -> String
    showLim x = show x

    sz :: [Int]
    sz = sizes tensor

-- | Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
inplaceFill
  :: (GenericMath t)
  => (a -> HaskReal t)
  -> a
  -> Ptr t
  -> IO ()
inplaceFill translate value = flip fill (translate value)

