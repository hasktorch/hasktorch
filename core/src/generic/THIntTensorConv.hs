{-# LANGUAGE ForeignFunctionInterface #-}

module THIntTensorConv (
    c_THIntTensor_validXCorr2Dptr,
    c_THIntTensor_validConv2Dptr,
    c_THIntTensor_fullXCorr2Dptr,
    c_THIntTensor_fullConv2Dptr,
    c_THIntTensor_validXCorr2DRevptr,
    c_THIntTensor_conv2DRevger,
    c_THIntTensor_conv2DRevgerm,
    c_THIntTensor_conv2Dger,
    c_THIntTensor_conv2Dmv,
    c_THIntTensor_conv2Dmm,
    c_THIntTensor_conv2Dmul,
    c_THIntTensor_conv2Dcmul,
    c_THIntTensor_validXCorr3Dptr,
    c_THIntTensor_validConv3Dptr,
    c_THIntTensor_fullXCorr3Dptr,
    c_THIntTensor_fullConv3Dptr,
    c_THIntTensor_validXCorr3DRevptr,
    c_THIntTensor_conv3DRevger,
    c_THIntTensor_conv3Dger,
    c_THIntTensor_conv3Dmv,
    c_THIntTensor_conv3Dmul,
    c_THIntTensor_conv3Dcmul) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_validXCorr2Dptr"
  c_THIntTensor_validXCorr2Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_validConv2Dptr"
  c_THIntTensor_validConv2Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_fullXCorr2Dptr"
  c_THIntTensor_fullXCorr2Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_fullConv2Dptr"
  c_THIntTensor_fullConv2Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_validXCorr2DRevptr"
  c_THIntTensor_validXCorr2DRevptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h THIntTensor_conv2DRevger"
  c_THIntTensor_conv2DRevger :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> IO ()

-- |c_THIntTensor_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h THIntTensor_conv2DRevgerm"
  c_THIntTensor_conv2DRevgerm :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> IO ()

-- |c_THIntTensor_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv2Dger"
  c_THIntTensor_conv2Dger :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv2Dmv"
  c_THIntTensor_conv2Dmv :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv2Dmm"
  c_THIntTensor_conv2Dmm :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv2Dmul"
  c_THIntTensor_conv2Dmul :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv2Dcmul"
  c_THIntTensor_conv2Dcmul :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_validXCorr3Dptr"
  c_THIntTensor_validXCorr3Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_validConv3Dptr"
  c_THIntTensor_validConv3Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_fullXCorr3Dptr"
  c_THIntTensor_fullXCorr3Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_fullConv3Dptr"
  c_THIntTensor_fullConv3Dptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THIntTensor_validXCorr3DRevptr"
  c_THIntTensor_validXCorr3DRevptr :: Ptr CInt -> CInt -> Ptr CInt -> CLong -> CLong -> CLong -> Ptr CInt -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall "THTensorConv.h THIntTensor_conv3DRevger"
  c_THIntTensor_conv3DRevger :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THIntTensor_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv3Dger"
  c_THIntTensor_conv3Dger :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv3Dmv"
  c_THIntTensor_conv3Dmv :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv3Dmul"
  c_THIntTensor_conv3Dmul :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THIntTensor_conv3Dcmul"
  c_THIntTensor_conv3Dcmul :: (Ptr CTHIntTensor) -> CInt -> CInt -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()