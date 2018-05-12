module Torch.Undefined.Tensor.Math.Random.TH where

import Foreign
import Foreign.C.Types
import Torch.Sig.Types
import Torch.Sig.Types.Global
import qualified Torch.Types.TH as TH

c_randperm :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CLLong -> IO ()
c_randperm = undefined
c_rand     :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr TH.CLongStorage -> IO ()
c_rand     = undefined
c_randn    :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr TH.CLongStorage -> IO ()
c_randn    = undefined


