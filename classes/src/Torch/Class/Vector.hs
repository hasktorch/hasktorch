module Torch.Class.Vector where

import Torch.Types.TH
import Foreign
import Foreign.C.Types
import Torch.Class.Internal

class Vector v where
  -- fill    :: v -> CTensor -> CPtrdiff -> IO ()
  -- cadd    :: v -> v -> v -> CTensor -> CPtrdiff -> IO ()
  -- adds    :: v -> v -> CTensor -> CPtrdiff -> IO ()
  -- cmul    :: v -> v -> v -> CPtrdiff -> IO ()
  -- muls    :: v -> v -> CTensor -> CPtrdiff -> IO ()
  -- cdiv    :: v -> v -> v -> CPtrdiff -> IO ()
  -- divs    :: v -> v -> CTensor -> CPtrdiff -> IO ()
  copy    :: v -> v -> CPtrdiff -> IO ()
  neg     :: v -> v -> CPtrdiff -> IO ()
  abs     :: v -> v -> CPtrdiff -> IO ()
  log     :: v -> v -> CPtrdiff -> IO ()
  lgamma  :: v -> v -> CPtrdiff -> IO ()
  log1p   :: v -> v -> CPtrdiff -> IO ()
  sigmoid :: v -> v -> CPtrdiff -> IO ()
  exp     :: v -> v -> CPtrdiff -> IO ()
  erf     :: v -> v -> CPtrdiff -> IO ()
  erfinv  :: v -> v -> CPtrdiff -> IO ()
  cos     :: v -> v -> CPtrdiff -> IO ()
  acos    :: v -> v -> CPtrdiff -> IO ()
  cosh    :: v -> v -> CPtrdiff -> IO ()
  sin     :: v -> v -> CPtrdiff -> IO ()
  asin    :: v -> v -> CPtrdiff -> IO ()
  sinh    :: v -> v -> CPtrdiff -> IO ()
  tan     :: v -> v -> CPtrdiff -> IO ()
  atan    :: v -> v -> CPtrdiff -> IO ()
  tanh    :: v -> v -> CPtrdiff -> IO ()
  -- pow     :: v -> v -> CTensor -> CPtrdiff -> IO ()
  sqrt    :: v -> v -> CPtrdiff -> IO ()
  rsqrt   :: v -> v -> CPtrdiff -> IO ()
  ceil    :: v -> v -> CPtrdiff -> IO ()
  floor   :: v -> v -> CPtrdiff -> IO ()
  round   :: v -> v -> CPtrdiff -> IO ()
  trunc   :: v -> v -> CPtrdiff -> IO ()
  frac    :: v -> v -> CPtrdiff -> IO ()
  cinv    :: v -> v -> CPtrdiff -> IO ()
