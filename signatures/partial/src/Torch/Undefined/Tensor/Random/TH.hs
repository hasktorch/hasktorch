module Torch.Undefined.Tensor.Random.TH where

import Foreign
import Foreign.C.Types
import Torch.Sig.Types
import Torch.Sig.Types.Global
import qualified Torch.Types.TH as TH

c_random                :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> IO ()
c_random                = undefined
c_clampedRandom         :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CLLong -> CLLong -> IO ()
c_clampedRandom         = undefined
c_cappedRandom          :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CLLong -> IO ()
c_cappedRandom          = undefined
c_geometric             :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> IO ()
c_geometric             = undefined
c_bernoulli             :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> IO ()
c_bernoulli             = undefined
c_bernoulli_FloatTensor :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr TH.CFloatTensor -> IO ()
c_bernoulli_FloatTensor = undefined
c_bernoulli_DoubleTensor      :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr TH.CDoubleTensor -> IO ()
c_bernoulli_DoubleTensor      = undefined

c_uniform              :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> CAccReal -> IO ()
c_uniform              = undefined
c_normal               :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> CAccReal -> IO ()
c_normal               = undefined
c_normal_means         :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr CTensor -> CAccReal -> IO ()
c_normal_means         = undefined
c_normal_stddevs       :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> Ptr CTensor -> IO ()
c_normal_stddevs       = undefined
c_normal_means_stddevs :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr CTensor -> Ptr CTensor -> IO ()
c_normal_means_stddevs = undefined
c_exponential          :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> IO ()
c_exponential          = undefined
c_standard_gamma       :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr CTensor -> IO ()
c_standard_gamma       = undefined
c_cauchy               :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> CAccReal -> IO ()
c_cauchy               = undefined
c_logNormal            :: Ptr CState -> Ptr CTensor -> Ptr CGenerator -> CAccReal -> CAccReal -> IO ()
c_logNormal            = undefined

c_multinomial           :: Ptr CState -> Ptr CLongTensor -> Ptr CGenerator -> Ptr CTensor -> CInt -> CInt -> IO ()
c_multinomial           = undefined
c_multinomialAliasSetup :: Ptr CState -> Ptr CTensor -> Ptr CLongTensor -> Ptr CTensor -> IO ()
c_multinomialAliasSetup = undefined
c_multinomialAliasDraw  :: Ptr CState -> Ptr CLongTensor -> Ptr CGenerator -> Ptr CLongTensor -> Ptr CTensor -> IO ()
c_multinomialAliasDraw  = undefined

