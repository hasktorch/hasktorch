module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Signed where

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Math.Pointwise.Signed as Sig

_abs r t = with2DynamicState r t Sig.c_abs
_neg r t = with2DynamicState r t Sig.c_neg

neg_, neg  :: Dynamic -> IO Dynamic
neg_ t = t `twice` _neg
neg  t = withEmpty t $ \r -> _neg r t

abs_, abs  :: Dynamic -> IO Dynamic
abs_ t = t `twice` _abs
abs  t = withEmpty t $ \r -> _abs r t


