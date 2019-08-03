-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Print
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Helper functions to render n-rank tensors
-------------------------------------------------------------------------------
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
module Torch.Indef.Dynamic.Print
  ( showTensor
  , describeTensor
  ) where

import Control.Applicative
import Control.Exception.Safe
import Control.Monad
import Data.List (intercalate)
import Data.List.NonEmpty (NonEmpty(..))
import Data.Maybe
import Data.Typeable
import GHC.Int
import GHC.Word
import Text.Printf

import qualified Data.List.NonEmpty as NE

import Torch.Indef.Types

-- | Generic way of showing the internal data of a tensor in a tabular format.
-- This makes no assumptions about the type of representation to show and can be
-- used for 'Storage', 'Dynamic', and 'Tensor' types.
showTensor
  :: forall a ix . (Typeable a, Ord a, Num a, Show a, Integral ix, Show ix)
  => (ix -> a)
  -> (ix -> ix -> a)
  -> (ix -> ix -> ix -> a)
  -> (ix -> ix -> ix -> ix -> a)
  -> [ix]
  -> String
showTensor get'1d get'2d get'3d get'4d ds =
  case ds of
    []  -> ""
    [x] -> brackets . intercalate "" $ fmap (valWithSpace . get'1d) (mkIx x)
    [x,y] -> go "" get'2d x y

    [z,x,y]   -> concat . flip fmap (mkIx z) $ \z' -> gt2IxHeader [z'] ++ (go "  " (get'3d z') x y)
    -- [x,y,z]   -> mat3dGo x y z
    [w,q,x,y] -> concat . flip fmap (mkXY w q) $ \(w', q') -> gt2IxHeader [w', q'] ++ (go "  " (get'4d w' q') x y)
    -- [x,y,z,q] -> mat4dGo x y z q

    _ -> error "Can't print this yet"
 where
  go :: String -> (ix -> ix -> a) -> ix -> ix -> String
  go fill getter x y = mat2dGo fill y "" $ fmap (valWithSpace . uncurry getter) (mkXY x y)

  mat2dGo :: String -> ix -> String -> [String] -> String
  mat2dGo    _ _ acc []  = acc
  mat2dGo fill y acc rcs = mat2dGo fill y acc' rest
    where
      (row, rest) = splitAt (fromIntegral y) rcs
      fullrow = fill ++ brackets (intercalate "" row)
      acc' = if null acc then fullrow else acc ++ "\n" ++ fullrow

  mat3dGo :: ix -> ix -> ix -> String
  mat3dGo x y z = concat $ flip fmap (mkIx x) $ \x' ->
    let mat = go "  " (get'3d x') y z
    in gt2IxHeader [x'] ++ mat

  mat4dGo :: ix -> ix -> ix -> ix -> String
  mat4dGo w q x y = concat $ flip fmap (mkXY w q) $ \(w', q') ->
    let mat = go "  " (get'4d w' q') x y
    in gt2IxHeader [w', q'] ++ mat


  mkIx :: ix -> [ix]
  mkIx x = [0..x - 1]

  mkXY :: ix -> ix -> [(ix, ix)]
  mkXY x y = [ (r, c) | r <- mkIx x, c <- mkIx y ]

  brackets :: String -> String
  brackets s = "[" ++ s ++ "]"

  valWithSpace :: (Typeable a, Ord a, Num a, Show a) => a -> String
  valWithSpace v = spacing ++ value ++ ""
   where
     truncTo :: (RealFrac x, Fractional x) => Int -> x -> x
     truncTo n f = fromInteger (round $ f * (10^n)) / (10.0^^n)

     value :: String
     value = fromMaybe (show v) $
           (printf "%.8f" <$> (cast v :: Maybe Double))
       <|> (printf "%.4f" <$> (cast v :: Maybe Float))

     spacing = magspacing ++ signspacing
     magspacing = ""
     -- magspacing = case compare (v `mod` 10) 4 of
     --   LT -> replicate (v `mod` 10)
     signspacing = case compare (signum v) 0 of
        LT -> " "
        _  -> "  "

  gt2IxHeader :: Show ix => [ix] -> String
  gt2IxHeader is = "\n(" ++ intercalate "," (fmap show is) ++ ",.,.):\n"

-- | show the shape of a tensor
describeTensor
  :: forall t dims
  . (Typeable t, Show dims)
  => [dims]
  -> Proxy t
  -> String
describeTensor ds t =
  "[" ++ show (typeRep t) ++ " tensor with shape: " ++ intercalate "x" (fmap show ds) ++ "]"

data TenSlices
  = TenNone
  | TenVector (NonEmpty HsReal)
  | TenMatricies (NonEmpty (NonEmpty [HsReal]))

-- | Helper function to show the matrix slices from a tensor.
tensorSlices
  :: Dynamic
  -> (Int64 -> IO HsReal)
  -> (Int64 -> Int64 -> IO HsReal)
  -- -> (Int64 -> Int64 -> Int64 -> IO HsReal)
  -- -> (Int64 -> Int64 -> Int64 -> Int64 -> IO HsReal)
  -> [Word64]
  -> IO TenSlices
tensorSlices t get'1d get'2d -- get'3d get'4d
  = \case
    []  -> pure TenNone
    [x] -> TenVector <$> go1d get'1d x
    [x,y] -> (TenMatricies . (:|[])) <$> go2d get'2d x y
    _ -> throwString "Can't slice this yet"
 where
  go1d :: (Int64 -> IO HsReal) -> Word64 -> IO (NonEmpty HsReal)
  go1d getter x
    = forM (mkIx x) getter

  go2d :: (Int64 -> Int64 -> IO HsReal) -> Word64 -> Word64 -> IO (NonEmpty [HsReal])
  go2d getter x y =
    forM (mkIx x) $ \ix ->
      forM (mkVIx y) $ \iy ->
        getter ix iy

  go3d :: (Int64 -> Int64 -> Int64 -> IO HsReal) -> Word64 -> Word64 -> Word64 -> IO (NonEmpty (NonEmpty [HsReal]))
  go3d getter x y z =
    forM (mkIx x) $ \ix ->
      forM (mkIx y) $ \iy ->
        -- forM [0..z - 1] $ \iz ->
          traverse (getter ix iy) (mkVIx z)

  -- mat2dGo :: Int64 -> String -> [HsReal] -> String
  -- mat2dGo _ acc []  = acc
  -- mat2dGo y acc rcs = mat2dGo y acc' rest
  --   where
  --     (row, rest) = splitAt (fromIntegral y) rcs
  --     acc' = if null acc then row else acc ++ "\n" ++ row

  -- mat3dGo :: Int64 -> Int64 -> Int64 -> IO String
  -- mat3dGo x y z = fmap (intercalate "") $ forM (mkIx x) $ \x' -> do
  --   mat <- go "  " (get'3d x') y z
  --   pure $ gt2IxHeader [x'] ++ mat

  -- mat4dGo :: Int64 -> Int64 -> Int64 -> Int64 -> IO String
  -- mat4dGo w q x y = fmap (intercalate "") $ forM (mkXY w q) $ \(w', q') -> do
  --   mat <- go "  " (get'4d w' q') x y
  --   pure $ gt2IxHeader [w', q'] ++ mat

  mkIx :: Word64 -> NonEmpty Int64
  mkIx 0 = 0 :| []
  mkIx x = 0 :| [1..fromIntegral x - 1]

  mkVIx :: Word64 -> [Int64]
  mkVIx 0 = []
  mkVIx x = [0..fromIntegral x - 1]


