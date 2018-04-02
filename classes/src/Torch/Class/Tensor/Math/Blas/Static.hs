module Torch.Class.Tensor.Math.Blas.Static where

import Torch.Class.Types

class TensorMathBlas t where
  addmv_       :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  addmm_       :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()

  -- outer product between a 1D tensor and a 1D tensor:
  -- https://github.com/torch/torch7/blob/aed31711c6b8846b8337a263a7f9f998697994e7/doc/maths.md#res-torchaddrres-v1-mat-v2-vec1-vec2
  --
  -- res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
  addr_        :: t '[r, c] -> HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO ()
  addbmm_      :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  baddbmm_     :: Dimensions4 d d' d'' d''' => t d -> HsReal (t d) -> t d' -> HsReal (t d) -> t d'' -> t d''' -> IO ()
  dot          :: Dimensions2 d d' => t d -> t d' -> IO (HsAccReal (t d))


-- addr :: (MathConstraint3 t '[r] '[c] '[r, c]) => HsReal (t '[r,c]) -> t '[r,c] -> HsReal (t '[r,c]) -> t '[r] -> t '[c] -> IO (t '[r, c])
-- addr  a t b x y = withInplace $ \r -> Dynamic.addr_ r a (asDynamic t) b (asDynamic x) (asDynamic y)

-- outer :: forall t r c . (MathConstraint3 t '[r] '[c] '[r, c]) => t '[r] -> t '[c] -> IO (t '[r, c])
-- outer v1 v2 = do
--   t :: t '[r, c] <- zerosLike
  -- addr 0 t 1 v1 v2


