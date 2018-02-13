module Torch.Core.Tensor.Static.Random.Floating where

import qualified Torch.Class.C.Tensor.Random as Class
import SigTypes (dynamic, Tensor)
import Torch.Core.Tensor.Dynamic.Random.Floating ()
import Torch.Core.Tensor.Static ()

instance Class.TensorRandomFloating (Tensor d) where
  uniform_ r = Class.uniform_ (dynamic r)
  normal_ r = Class.normal_ (dynamic r)
  normal_means_ r g m = Class.normal_means_ (dynamic r) g (dynamic m)
  normal_stddevs_ r g m s = Class.normal_stddevs_ (dynamic r) g m (dynamic s)
  normal_means_stddevs_ r g m s = Class.normal_means_stddevs_ (dynamic r) g (dynamic m) (dynamic s)
  exponential_ r = Class.exponential_ (dynamic r)
  standard_gamma_ r g m = Class.standard_gamma_ (dynamic r) g (dynamic m)
  cauchy_ r = Class.cauchy_ (dynamic r)
  logNormal_ r = Class.logNormal_ (dynamic r)

--  multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> t -> Int32 -> Int32 -> IO ()
--  multinomialAliasSetup  :: t -> Ptr CTHLongTensor -> t -> IO ()
--  multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> t -> IO ()


