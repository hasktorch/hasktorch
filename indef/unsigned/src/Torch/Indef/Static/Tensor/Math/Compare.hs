module Torch.Indef.Static.Tensor.Math.Compare where

class TensorMathCompare t where
  ltValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  ltValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_ltValue bt' t0' (hs2cReal v)

  leValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  leValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_leValue bt' t0' (hs2cReal v)

  gtValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  gtValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_gtValue bt' t0' (hs2cReal v)

  geValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  geValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_geValue bt' t0' (hs2cReal v)

  neValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  neValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_neValue bt' t0' (hs2cReal v)

  eqValue_ :: ByteTensor -> Tensor -> HsReal -> IO ()
  eqValue_ bt t0 v = _withTensor t0 $ \t0' ->withForeignPtr (B.tensor bt) $ \bt' ->  Sig.c_eqValue bt' t0' (hs2cReal v)

  ltValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  ltValueT_ = twoTensorsAndReal Sig.c_ltValueT

  leValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  leValueT_ = twoTensorsAndReal Sig.c_leValueT

  gtValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  gtValueT_ = twoTensorsAndReal Sig.c_gtValueT

  geValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  geValueT_ = twoTensorsAndReal Sig.c_geValueT

  neValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  neValueT_ = twoTensorsAndReal Sig.c_neValueT

  eqValueT_ :: Tensor -> Tensor -> HsReal -> IO ()
  eqValueT_ = twoTensorsAndReal Sig.c_eqValueT


