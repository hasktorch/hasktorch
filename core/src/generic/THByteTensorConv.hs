{-# LANGUAGE ForeignFunctionInterface #-}

module THByteTensorConv (
    c_THByteTensor_validXCorr2Dptr,
    c_THByteTensor_validConv2Dptr,
    c_THByteTensor_fullXCorr2Dptr,
    c_THByteTensor_fullConv2Dptr,
    c_THByteTensor_validXCorr2DRevptr,
    c_THByteTensor_conv2DRevger,
    c_THByteTensor_conv2DRevgerm,
    c_THByteTensor_conv2Dger,
    c_THByteTensor_conv2Dmv,
    c_THByteTensor_conv2Dmm,
    c_THByteTensor_conv2Dmul,
    c_THByteTensor_conv2Dcmul,
    c_THByteTensor_validXCorr3Dptr,
    c_THByteTensor_validConv3Dptr,
    c_THByteTensor_fullXCorr3Dptr,
    c_THByteTensor_fullConv3Dptr,
    c_THByteTensor_validXCorr3DRevptr,
    c_THByteTensor_conv3DRevger,
    c_THByteTensor_conv3Dger,
    c_THByteTensor_conv3Dmv,
    c_THByteTensor_conv3Dmul,
    c_THByteTensor_conv3Dcmul) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_validXCorr2Dptr"
  c_THByteTensor_validXCorr2Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_validConv2Dptr"
  c_THByteTensor_validConv2Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_fullXCorr2Dptr"
  c_THByteTensor_fullXCorr2Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_fullConv2Dptr"
  c_THByteTensor_fullConv2Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_validXCorr2DRevptr"
  c_THByteTensor_validXCorr2DRevptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv2DRevger"
  c_THByteTensor_conv2DRevger :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> IO ()

-- |c_THByteTensor_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv2DRevgerm"
  c_THByteTensor_conv2DRevgerm :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> IO ()

-- |c_THByteTensor_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv2Dger"
  c_THByteTensor_conv2Dger :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv2Dmv"
  c_THByteTensor_conv2Dmv :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv2Dmm"
  c_THByteTensor_conv2Dmm :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv2Dmul"
  c_THByteTensor_conv2Dmul :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv2Dcmul"
  c_THByteTensor_conv2Dcmul :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_validXCorr3Dptr"
  c_THByteTensor_validXCorr3Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_validConv3Dptr"
  c_THByteTensor_validConv3Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_fullXCorr3Dptr"
  c_THByteTensor_fullXCorr3Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_fullConv3Dptr"
  c_THByteTensor_fullConv3Dptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_validXCorr3DRevptr"
  c_THByteTensor_validXCorr3DRevptr :: Ptr CChar -> CChar -> Ptr CChar -> CLong -> CLong -> CLong -> Ptr CChar -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv3DRevger"
  c_THByteTensor_conv3DRevger :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THByteTensor_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv3Dger"
  c_THByteTensor_conv3Dger :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv3Dmv"
  c_THByteTensor_conv3Dmv :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv3Dmul"
  c_THByteTensor_conv3Dmul :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THByteTensor_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THByteTensor_conv3Dcmul"
  c_THByteTensor_conv3Dcmul :: (Ptr CTHByteTensor) -> CChar -> CChar -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()