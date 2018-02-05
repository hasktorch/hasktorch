signature Blas where

data CTensor
data CReal
data CAccReal

c_swap :: CLLong -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> IO ()
c_scal :: CLLong -> CTensor -> Ptr CTensor -> CLLong -> IO ()
c_copy :: CLLong -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> IO ()
c_axpy :: CLLong -> CTensor -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> IO ()
c_dot  :: CLLong -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> CTensor
c_gemv :: CChar -> CLLong -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> IO ()
c_ger  :: CLLong -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> PtrCTensor-> CLLong -> IO ()

p_swap :: FunPtr (CLLong -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> IO ())
p_scal :: FunPtr (CLLong -> CTensor -> Ptr CTensor -> CLLong -> IO ())
p_copy :: FunPtr (CLLong -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> IO ())
p_axpy :: FunPtr (CLLong -> CTensor -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> IO ())
p_dot  :: FunPtr (CLLong -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> t)
p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> IO ())
p_ger  :: FunPtr (CLLong -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> IO ())
p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> Ptr CTensor -> CLLong -> CTensor -> Ptr CTensor -> CLLong -> IO ())

