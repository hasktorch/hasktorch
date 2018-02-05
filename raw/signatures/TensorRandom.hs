signature Torch.Raw.Tensor.Random where

data CTensor
data CReal
data CAccReal

c_random                 :: Ptr CTensor -> Ptr CTHGenerator -> IO ()
c_clampedRandom          :: Ptr CTensor -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()
c_cappedRandom           :: Ptr CTensor -> Ptr CTHGenerator -> CLLong -> IO ()
c_geometric              :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> IO ()
c_bernoulli              :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> IO ()
c_bernoulli_FloatTensor  :: Ptr CTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()
c_bernoulli_DoubleTensor :: Ptr CTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()
c_uniform                :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> CAccReal -> IO ()
c_normal                 :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> CAccReal -> IO ()
c_normal_means           :: Ptr CTensor -> Ptr CTHGenerator -> Ptr CTensor -> CAccReal -> IO ()
c_normal_stddevs         :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> Ptr CTensor -> IO ()
c_normal_means_stddevs   :: Ptr CTensor -> Ptr CTHGenerator -> Ptr CTensor -> Ptr CTensor -> IO ()
c_exponential            :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> IO ()
c_standard_gamma         :: Ptr CTensor -> Ptr CTHGenerator -> Ptr CTensor -> IO ()
c_cauchy                 :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> CAccReal -> IO ()
c_logNormal              :: Ptr CTensor -> Ptr CTHGenerator -> CAccReal -> CAccReal -> IO ()
c_multinomial            :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTensor -> CInt -> CInt -> IO ()
c_multinomialAliasSetup  :: Ptr CTensor -> Ptr CTHLongTensor -> Ptr CTensor -> IO ()
c_multinomialAliasDraw   :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Ptr CTensor -> IO ()
