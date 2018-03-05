signature Torch.Signature.Signed.Tensor.Math where

import Torch.Types.TH
import Foreign
import Foreign.C.Types
import Torch.Signature.Types

c_neg :: Ptr CTensor -> Ptr CTensor -> IO ()
c_abs :: Ptr CTensor -> Ptr CTensor -> IO ()


