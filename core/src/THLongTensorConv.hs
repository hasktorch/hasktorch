{-# LANGUAGE ForeignFunctionInterface#-}

module THLongTensorConv (
    c_THLongTensor_validXCorr2Dptr,
    c_THLongTensor_validConv2Dptr,
    c_THLongTensor_fullXCorr2Dptr,
    c_THLongTensor_fullConv2Dptr,
    c_THLongTensor_validXCorr2DRevptr,
    c_THLongTensor_conv2DRevger,
    c_THLongTensor_conv2DRevgerm,
    c_THLongTensor_conv2Dger,
    c_THLongTensor_conv2Dmv,
    c_THLongTensor_conv2Dmm,
    c_THLongTensor_conv2Dmul,
    c_THLongTensor_conv2Dcmul,
    c_THLongTensor_validXCorr3Dptr,
    c_THLongTensor_validConv3Dptr,
    c_THLongTensor_fullXCorr3Dptr,
    c_THLongTensor_fullConv3Dptr,
    c_THLongTensor_validXCorr3DRevptr,
    c_THLongTensor_conv3DRevger,
    c_THLongTensor_conv3Dger,
    c_THLongTensor_conv3Dmv,
    c_THLongTensor_conv3Dmul,
    c_THLongTensor_conv3Dcmul) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_validXCorr2Dptr"
  c_THLongTensor_validXCorr2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_validConv2Dptr"
  c_THLongTensor_validConv2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_fullXCorr2Dptr"
  c_THLongTensor_fullXCorr2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_fullConv2Dptr"
  c_THLongTensor_fullConv2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_validXCorr2DRevptr"
  c_THLongTensor_validXCorr2DRevptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h THLongTensor_conv2DRevger"
  c_THLongTensor_conv2DRevger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h THLongTensor_conv2DRevgerm"
  c_THLongTensor_conv2DRevgerm :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv2Dger"
  c_THLongTensor_conv2Dger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv2Dmv"
  c_THLongTensor_conv2Dmv :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv2Dmm"
  c_THLongTensor_conv2Dmm :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv2Dmul"
  c_THLongTensor_conv2Dmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv2Dcmul"
  c_THLongTensor_conv2Dcmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_validXCorr3Dptr"
  c_THLongTensor_validXCorr3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_validConv3Dptr"
  c_THLongTensor_validConv3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_fullXCorr3Dptr"
  c_THLongTensor_fullXCorr3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_fullConv3Dptr"
  c_THLongTensor_fullConv3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h THLongTensor_validXCorr3DRevptr"
  c_THLongTensor_validXCorr3DRevptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall "THTensorConv.h THLongTensor_conv3DRevger"
  c_THLongTensor_conv3DRevger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv3Dger"
  c_THLongTensor_conv3Dger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv3Dmv"
  c_THLongTensor_conv3Dmv :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv3Dmul"
  c_THLongTensor_conv3Dmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CChar -> CChar -> IO ()

-- |c_THLongTensor_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h THLongTensor_conv3Dcmul"
  c_THLongTensor_conv3Dcmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> CChar -> CChar -> IO ()