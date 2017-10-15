module HMatrixBridge where

import Data.Maybe (fromJust)
import Numeric.LinearAlgebra hiding (size, disp)
-- import TorchTensor

import THDoubleTensor
import THDoubleTensorMath

import Foreign
import Foreign.C.Types

-- t2list :: TensorDouble -> [Double]
-- t2list t = do
--   fmap (\idx -> ((realToFrac $ c_THDoubleTensor_get1d t idx) :: Double)) indexes
--   where
--     sz = size t
--     indexes = [ fromIntegral idx :: CLong
--               | idx <- [0..(sz !! 0 - 1)] ]

-- t2vector = vector . t2list

-- test1 = do
--   vec <- fromJust $ tensorNew [3]
--   c_THDoubleTensor_fill vec 2.0
--   print $ t2vector vec
