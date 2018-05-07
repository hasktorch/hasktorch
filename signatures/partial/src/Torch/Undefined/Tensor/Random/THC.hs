module Torch.Undefined.Tensor.Random.THC where

import Foreign
import Foreign.C.Types
import Torch.Sig.Types
import Torch.Sig.Types.Global
import qualified Torch.Types.TH as TH (CLongStorage)

c_random :: Ptr CState -> Ptr CTensor -> IO ()
c_random = undefined
c_clampedRandom :: Ptr CState -> Ptr CTensor -> CLLong -> CLLong -> IO ()
c_clampedRandom = undefined
c_cappedRandom :: Ptr CState -> Ptr CTensor -> CLLong -> IO ()
c_cappedRandom = undefined
c_bernoulli :: Ptr CState -> Ptr CTensor -> CAccReal -> IO ()
c_bernoulli = undefined
c_bernoulli_DoubleTensor :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_bernoulli_DoubleTensor = undefined
c_geometric :: Ptr CState -> Ptr CTensor -> CAccReal -> IO ()
c_geometric = undefined

c_uniform :: Ptr CState -> Ptr CTensor -> CAccReal -> CAccReal -> IO ()
c_uniform = undefined
c_normal :: Ptr CState -> Ptr CTensor -> CAccReal -> CAccReal -> IO ()
c_normal = undefined
c_normal_means :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CAccReal -> IO ()
c_normal_means = undefined
c_normal_stddevs :: Ptr CState -> Ptr CTensor -> CAccReal -> Ptr CTensor -> IO ()
c_normal_stddevs = undefined
c_normal_means_stddevs :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_normal_means_stddevs = undefined
c_logNormal :: Ptr CState -> Ptr CTensor -> CAccReal -> CAccReal -> IO ()
c_logNormal = undefined
c_exponential :: Ptr CState -> Ptr CTensor -> CAccReal -> IO ()
c_exponential = undefined
c_cauchy :: Ptr CState -> Ptr CTensor -> CAccReal -> CAccReal -> IO ()
c_cauchy = undefined

c_multinomial :: Ptr CState -> Ptr CIndexTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_multinomial = undefined
c_multinomialAliasSetup :: Ptr CState -> Ptr CTensor -> Ptr CIndexTensor -> Ptr CTensor -> IO ()
c_multinomialAliasSetup = undefined
c_multinomialAliasDraw :: Ptr CState -> Ptr CIndexTensor -> Ptr CIndexTensor -> Ptr CTensor -> IO ()
c_multinomialAliasDraw = undefined

c_rand :: Ptr CState -> Ptr CTensor -> Ptr TH.CLongStorage -> IO ()
c_rand = undefined
c_randn :: Ptr CState -> Ptr CTensor -> Ptr TH.CLongStorage -> IO ()
c_randn = undefined

