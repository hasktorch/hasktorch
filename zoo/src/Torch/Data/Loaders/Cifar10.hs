{-# LANGUAGE CPP #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
module Torch.Data.Loaders.Cifar10
  ( default_cifar_path
  , Mode(..)
  , mode_path
  , testLength
  , trainLength
  , Category(..)
  , I.rgb2torch
  , cifar10set
  , defaultCifar10set
  ) where

import System.FilePath ((</>))
import Data.Proxy (Proxy(..))
import GHC.Generics (Generic)
import Text.Read (readMaybe)
import Control.DeepSeq (NFData)
import Data.Vector (Vector)
import System.Random.MWC (GenIO, createSystemRandom)

#ifdef CUDA
import Torch.Cuda.Double
#else
import Torch.Double
#endif

import qualified Data.Char as Char
import qualified Torch.Data.Loaders.Internal as I

-- This should be replaced with a download-aware cache.
default_cifar_path :: FilePath
default_cifar_path = "/mnt/lake/datasets/cifar-10"

data Mode = Test | Train
  deriving (Eq, Enum, Ord, Show, Bounded)

testLength  :: Proxy 'Test -> Proxy 1000
testLength _ = Proxy

trainLength :: Proxy 'Train -> Proxy 5000
trainLength _ = Proxy

data Category
  = Airplane    -- 0
  | Automobile  -- 2
  | Bird        -- 3
  | Cat         -- 4
  | Deer        -- 5
  | Dog         -- 6
  | Frog        -- 7
  | Horse       -- 8
  | Ship        -- 9
  | Truck       -- 10
  deriving (Eq, Enum, Ord, Show, Bounded, Generic, NFData, Read)

mode_path :: FilePath -> Mode -> FilePath
mode_path cifarpath m = cifarpath </> (Char.toLower <$> show m)

cifar10set :: GenIO -> FilePath -> Mode -> IO (Vector (Category, FilePath))
cifar10set g p m = I.shuffleCatFolders g cast (mode_path p m)
 where
  cast :: FilePath -> Maybe Category
  cast fp =
    case filter (not . (`elem` ("/\\"::String))) fp of
      h:tl -> readMaybe (Char.toUpper h : map Char.toLower tl)
      _    -> Nothing

defaultCifar10set :: Mode -> IO (Vector (Category, FilePath))
defaultCifar10set m =
  createSystemRandom >>= \g -> cifar10set g default_cifar_path m

-- test :: Tensor '[1]
-- test
--   = evalBP
--       (classNLLCriterion (Long.unsafeVector [2] :: Long.Tensor '[1]))
--       (unsqueeze1d (dim :: Dim 0) $ unsafeVector [1,0,0] :: Tensor '[1, 3])
--
-- test2 :: Tensor '[1]
-- test2
--   = evalBP
--   ( _classNLLCriterion'
--       (-100) False True
--       -- (Long.unsafeMatrix [[0,1,0]] :: Long.Tensor '[1,3])
--       -- (Long.unsafeVector [0,1,0] :: Long.Tensor '[3])
--       (Long.unsafeVector [0,1,2] :: Long.Tensor '[3])
--     )
--     -- (unsafeVector  [1,0,0]  :: Tensor '[3])
--     -- (unsafeMatrix [[0,0,1]] :: Tensor '[1,3])
--     (unsafeMatrix
--       [ [1,0,0]
--       , [0,1,0]
--       , [0.5,0.5,0.5]
--       ] :: Tensor '[3,3])

-- test3 :: CPU.Tensor '[1]
-- test3
--   = evalBP
--   ( CPU._classNLLCriterion'
--       (-100) False True
--       -- (CPULong.unsafeMatrix [[0,1,0]] :: CPULong.Tensor '[1,3])
--       (CPULong.unsafeVector [0,8] :: CPULong.Tensor '[2])
--       -- (CPULong.unsafeVector [0,1,0] :: CPULong.Tensor '[3])
--       -- (CPULong.unsafeVector [0,1,2] :: CPULong.Tensor '[3])
--     )
--     -- (CPU.unsafeVector  [1,0,0]  :: CPU.Tensor '[3])
--     -- (CPU.unsafeMatrix [[0,0,1]] :: CPU.Tensor '[1,3])
--     (CPU.unsafeMatrix
--       [ [1,0,0]
--       -- , [0,1,0]
--       , [0.5,0.5,0.5]
--       ] :: CPU.Tensor '[2,3])


