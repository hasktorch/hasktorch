module Torch.Class.Tensor.Math.Blas.Static where

import Torch.Class.Types

class TensorMathBlas t where
  addmv_       :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  addmm_       :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  addr_        :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  addbmm_      :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  baddbmm_     :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  dot          :: Dimensions2 d d' => t d -> t d' -> IO (HsAccReal (t d))


