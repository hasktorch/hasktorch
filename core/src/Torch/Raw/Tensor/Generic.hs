{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Raw.Tensor.Generic
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
  , genericEqual
  , genericNew
  , genericNew'
  , genericGet
  , genericGet'
  , genericInvLogit

  , dispRaw
  , dispRawRealFloat
  , minLogScale
  , numberFormat

  , module X
  ) where

import Numeric.Dimensions (Dim(..), someDimsVal)
import qualified Numeric.Limits as NL
import Foreign (Ptr)
import Foreign.C.Types
import Data.Maybe (fromJust)
import Data.Fixed (mod')
import qualified Numeric as Num (showGFloat)
import Text.Printf (printf)

import Torch.Core.Internal (impossible)
import Torch.Core.Tensor.Dim

import Torch.Raw.Internal as X
import Torch.Raw.Tensor as X
import Torch.Raw.Tensor.Math as X
import Torch.Raw.Tensor.Lapack as X
import Torch.Raw.Tensor.Random as X

import THTypes

-- | flatten a CTHDoubleTensor into a list
flatten :: THTensor t => Ptr t -> [HaskReal t]
flatten tensor =
  case map getDim [0 .. c_nDimension tensor - 1] of
    []           -> mempty
    [x]          -> c_get1d tensor <$> range x
    [x, y]       -> c_get2d tensor <$> range x <*> range y
    [x, y, z]    -> c_get3d tensor <$> range x <*> range y <*> range z
    [x, y, z, q] -> c_get4d tensor <$> range x <*> range y <*> range z <*> range q
    _ -> error "TH doesn't support getting tensors higher than 4-dimensions"
  where
    getDim :: CInt -> Int
    getDim = fromIntegral . c_size tensor

    range :: Integral i => Int -> [i]
    range mx = [0 .. fromIntegral mx - 1]

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInit
  :: (THTensorMath t, THTensorRandom t, THTensor t, Num (HaskReal t))
  => Ptr CTHGenerator
  -> Dim (dims :: [k])
  -> HaskAccReal t
  -> HaskAccReal t
  -> IO (Ptr t)
randInit gen dims lower upper = do
  t <- constant dims 0
  c_uniform t gen lower upper
  pure t

-- |randomly initialize a tensor with uniform random values from a range
-- TODO - finish implementation to handle sizes correctly
randInit'
  :: (THTensorMath t, THTensorRandom t, THTensor t, Num (HaskReal t))
  => Ptr CTHGenerator
  -> SomeDims
  -> HaskAccReal t
  -> HaskAccReal t
  -> IO (Ptr t)
randInit' gen dims lower upper = do
  t <- constant' dims 0
  c_uniform t gen lower upper
  pure t

-- | Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
-- fillDouble :: (THTensorMath t, THTensor t) => HaskReal t -> Ptr t -> IO ()
-- fillDouble = flip c_fill . realToFrac

-- | Create a new (double) tensor of specified dimensions and c_fill it with 0
-- safe version
-- tensorRaw :: Dim (ns::[k]) -> Double -> IO TensorDoubleRaw
constant :: forall ns t . (THTensorMath t, THTensor t) => Dim (ns::[k]) -> HaskReal t -> IO (Ptr t)
constant dims value = do
  newPtr <- genericNew dims
  c_fill newPtr value
  pure newPtr

-- | Create a new (double) tensor of specified dimensions and c_fill it with 0
-- safe version
constant' :: forall ns t . (THTensorMath t, THTensor t) => SomeDims -> HaskReal t -> IO (Ptr t)
constant' dims value = do
  newPtr <- genericNew' dims
  c_fill newPtr value
  pure newPtr


-- | Generic equality
genericEqual :: THTensorMath t => Ptr t -> Ptr t -> Bool
genericEqual a b = 1 == c_equal a b


genericNew :: THTensor t => Dim (ns::[k]) -> IO (Ptr t)
genericNew = onDims fromIntegral
  c_new
  c_newWithSize1d
  c_newWithSize2d
  c_newWithSize3d
  c_newWithSize4d

genericNew' :: THTensor t => SomeDims -> IO (Ptr t)
genericNew' = onDims' fromIntegral
  c_new
  c_newWithSize1d
  c_newWithSize2d
  c_newWithSize3d
  c_newWithSize4d

genericGet :: THTensor t => Ptr t -> Dim (ns::[k]) -> HaskReal t
genericGet t = onDims fromIntegral
  (impossible "0-rank will never be called")
  (c_get1d t)
  (c_get2d t)
  (c_get3d t)
  (c_get4d t)

genericGet' :: THTensor t => Ptr t -> SomeDims -> HaskReal t
genericGet' t = onDims' fromIntegral
  (impossible "0-rank will never be called")
  (c_get1d t)
  (c_get2d t)
  (c_get3d t)
  (c_get4d t)


-- | apply inverse logit to all values of a tensor
genericInvLogit :: (THTensorMathFloating t, THTensor t) => Ptr t -> IO (Ptr t)
genericInvLogit = applyInPlaceFn c_sigmoid


-- |apply a tensor transforming function to a tensor
applyInPlaceFn :: THTensor t => (Ptr t -> Ptr t -> IO ()) -> Ptr t -> IO (Ptr t)
applyInPlaceFn f t1 = do
  r_ <- c_new
  f r_ t1
  pure r_

-- | Dimensions of a raw tensor as a list
dimList :: THTensor t => Ptr t -> [Int]
dimList t = getDim <$> [0 .. c_nDimension t - 1]
  where
    getDim :: CInt -> Int
    getDim = fromIntegral . c_size t

-- |Dimensions of a raw tensor as a TensorDim value
dimView :: THTensor t => Ptr t -> DimView
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
getDynamicDim :: THTensor t => Ptr t -> SomeDims
getDynamicDim
-- Note: we can safely call 'fromJust' since these values are maintained by TH which does the bounds-checking
  = fromJust . someDimsVal . dimList

-- | c_fill a raw Double tensor with 0.0
fillZeros :: (THTensorMath t, THTensor t, Num (HaskReal t)) => Ptr t -> IO (Ptr t)
fillZeros t = c_fill t 0 >> pure t

-- Display stuff
minLogScale :: Int
minLogScale =
    ceiling $ logBase 10 p
  where
    minPositive = NL.minValue * NL.epsilon  -- get smallest denormal
    p = if (minPositive == 0) then NL.minValue else minPositive

numberFormat :: RealFloat a => Int -> Int -> [a] -> (String, a, Int)
numberFormat precision minSz xs = (format, scale, sz)
  where
    listParams :: (RealFloat a) => [a] -> (a, a, Bool, Bool, Bool)
    listParams z =
      f (NL.infinity, -NL.infinity, True, False, False) z
      where
        f z [] = z
        f (minAcc, maxAcc, intModeAcc, hasInvalidAcc, hasNegInfAcc) (x:xs) = 
            f acc xs
          where
            val = abs x
            newHasNegInf = (x == -NL.infinity) || hasNegInfAcc
            acc =
              if (isNaN val || val == NL.infinity) then
                (minAcc, maxAcc, intModeAcc, True, newHasNegInf) 
              else 
                (newMin, newMax, newIntMode, newHasInvalid, newHasNegInf)
                  where
                    !newHasInvalid = hasInvalidAcc
                    !newMin = min minAcc val
                    !newMax = max maxAcc val
                    !newIntMode = if (intModeAcc) then
                                    if (val == NL.infinity) then intModeAcc
                                    else mod' val 1 == 0
                                  else 
                                    False
    prec = precision
    defaultScale = 1.0
    (minAbs, maxAbs, intMode, hasInvalid, hasNegInf) = listParams xs
    expMin = if (minAbs > 0.0) then 1 + (floor $ logBase 10 minAbs) else 1
    expMax = if (maxAbs > 0.0) then 1 + (floor $ logBase 10 maxAbs) else 1
    myMinSz = if hasInvalid then max minSz $ 3 + (if hasNegInf then 1 else 0)
              else max minSz 2
    (format, scale, sz) =
      let expSz = (prec + 4 + (max (length (show expMax)) 
                                   (length (show expMin)))) in
      if (intMode) then
        if (expMax > prec + 1) then
          let tmpSz = max myMinSz expSz in
          let format = "%" ++ show tmpSz ++ "." ++ show prec ++ "e" in
          (format, defaultScale, tmpSz)
        else
          let tmpSz = max myMinSz (expMax + 1) in
          let format = "%" ++ show tmpSz ++ ".0f" in
          (format, defaultScale, tmpSz) 
      else
        if (expMax - expMin > prec) then
          let tmpSz = max myMinSz expSz in
          let format = "%" ++ show tmpSz ++ "." ++ show prec ++ "e" in
          (format, 1.0, tmpSz) 
        else
          if (expMax > prec + 1) || (expMax < 0) then
            let tmpSz = max myMinSz 7 in
            let format = "%" ++ show tmpSz ++ "." ++ show prec ++ "f" in
            (format, 10.0**fromIntegral(max minLogScale (expMax - 1)), tmpSz)
          else
            let tmpSz = if (expMax > 0) then expMax + 6 else 7 in
            let format = "%" ++ show tmpSz ++ "." ++ show prec ++ "f" in
            (format, 1.0, max myMinSz tmpSz)


dispRawIntegral tensor = undefined

dispRawRealFloat :: (THTensor t, RealFloat (HaskReal t)) => Ptr t -> IO ()
dispRawRealFloat tensor
 | length sz == 0 = putStrLn "Empty Tensor"
 | length sz == 1 = do
     putStrLn ""
     let indexes = [ fromIntegral idx :: CLLong
                   | idx <- [0..head sz - 1]
                   ]
     putStr "[ "
     mapM_ (\idx -> putStr $ showLim (c_get1d tensor idx) ++ " ")
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
               let val = c_get2d tensor r c
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
   sizes :: (THTensor t) => Ptr t -> [Int]
   sizes t = fmap (fromIntegral . c_size t) [0..c_nDimension t - 1]

   sz :: [Int]
   sz = sizes tensor

   (fmt, scale, size) = numberFormat 4 3 $ flatten tensor

   showLim :: RealFloat a => a -> String
   showLim val = 
     if (val == NL.infinity) then 
       Text.Printf.printf ("%"++show size++"s") "Inf"
     else if (val == -NL.infinity) then 
       Text.Printf.printf ("%"++show size++"s") "-Inf"
     else Text.Printf.printf fmt (((realToFrac val)::Double)/realToFrac(scale))
  

-- ========================================================================= --
-- TO BE REMOVED: dispRaw
-- ========================================================================= --
--
-- | displaying raw tensor values
dispRaw :: forall t . (THTensor t, Show (HaskReal t)) => Ptr t -> IO ()
dispRaw tensor
 | length sz == 0 = putStrLn "Empty Tensor"
 | length sz == 1 = do
     putStrLn ""
     let indexes = [ fromIntegral idx :: CLLong
                   | idx <- [0..head sz - 1]
                   ]
     putStr "[ "
     mapM_ (\idx -> putStr $ showLim (c_get1d tensor idx) ++ " ")
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
               let val = c_get2d tensor r c
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
   sizes t = fmap (fromIntegral . c_size t) [0..c_nDimension t - 1]

   showLim :: HaskReal t -> String
   showLim x = show x

   sz :: [Int]
   sz = sizes tensor

-- | Returns a function that accepts a tensor and fills it with specified value
-- and returns the IO context with the mutated tensor
--
-- renamed from @fillRaw :: Real a => a -> TensorDoubleRaw -> IO ()@
inplaceFill
  :: (THTensorMath t)
  => (a -> HaskReal t)
  -> a
  -> Ptr t
  -> IO ()
inplaceFill translate value = flip c_fill (translate value)

