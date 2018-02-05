signatures Torch.Raw.Storage where

data CTensor
data CReal
data CAccReal

c_data                    :: Ptr CTensor -> IO (Ptr CReal)
c_size                    :: Ptr CTensor -> CPtrdiff
-- c_elementSize          :: CSize
c_set                     :: Ptr CTensor -> CPtrdiff -> CReal -> IO ()
c_get                     :: Ptr CTensor -> CPtrdiff -> CReal
c_new                     :: IO (Ptr CTensor)
c_newWithSize             :: CPtrdiff -> IO (Ptr CTensor)
c_newWithSize1            :: CReal -> IO (Ptr CTensor)
c_newWithSize2            :: CReal -> CReal -> IO (Ptr CTensor)
c_newWithSize3            :: CReal -> CReal -> CReal -> IO (Ptr CTensor)
c_newWithSize4            :: CReal -> CReal -> CReal -> CReal -> IO (Ptr CTensor)
c_newWithMapping          :: Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTensor)
c_newWithData             :: Ptr CReal -> CPtrdiff -> IO (Ptr CTensor)
c_newWithAllocator        :: CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTensor)
c_newWithDataAndAllocator :: Ptr CReal -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTensor)
c_setFlag                 :: Ptr CTensor -> CChar -> IO ()
c_clearFlag               :: Ptr CTensor -> CChar -> IO ()
c_retain                  :: Ptr CTensor -> IO ()
c_swap                    :: Ptr CTensor -> Ptr CTensor -> IO ()
c_free                    :: Ptr CTensor -> IO ()
c_resize                  :: Ptr CTensor -> CPtrdiff -> IO ()
c_fill                    :: Ptr CTensor -> CReal -> IO ()

p_data                    :: FunPtr (Ptr CTensor -> IO (Ptr CReal))
p_size                    :: FunPtr (Ptr CTensor -> CPtrdiff)
-- p_elementSize          :: FunPtr CSize
p_set                     :: FunPtr (Ptr CTensor -> CPtrdiff -> CReal -> IO ())
p_get                     :: FunPtr (Ptr CTensor -> CPtrdiff -> CReal)
p_new                     :: FunPtr (IO (Ptr CTensor))
p_newWithSize             :: FunPtr (CPtrdiff -> IO (Ptr CTensor))
p_newWithSize1            :: FunPtr (CReal -> IO (Ptr CTensor))
p_newWithSize2            :: FunPtr (CReal -> CReal -> IO (Ptr CTensor))
p_newWithSize3            :: FunPtr (CReal -> CReal -> CReal -> IO (Ptr CTensor))
p_newWithSize4            :: FunPtr (CReal -> CReal -> CReal -> CReal -> IO (Ptr CTensor))
p_newWithMapping          :: FunPtr (Ptr CChar -> CPtrdiff -> CInt -> IO (Ptr CTensor))
p_newWithData             :: FunPtr (Ptr CReal -> CPtrdiff -> IO (Ptr CTensor))
p_newWithAllocator        :: FunPtr (CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTensor))
p_newWithDataAndAllocator :: FunPtr (Ptr CReal -> CPtrdiff -> CTHAllocatorPtr -> Ptr () -> IO (Ptr CTensor))
p_setFlag                 :: FunPtr (Ptr CTensor -> CChar -> IO ())
p_clearFlag               :: FunPtr (Ptr CTensor -> CChar -> IO ())
p_retain                  :: FunPtr (Ptr CTensor -> IO ())
p_swap                    :: FunPtr (Ptr CTensor -> Ptr CTensor -> IO ())
p_free                    :: FunPtr (Ptr CTensor -> IO ())
p_resize                  :: FunPtr (Ptr CTensor -> CPtrdiff -> IO ())
p_fill                    :: FunPtr (Ptr CTensor -> CReal -> IO ())
