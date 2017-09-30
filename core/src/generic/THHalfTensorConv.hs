{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfTensorConv (
    c_THHalfTensor_validXCorr2Dptr,
    c_THHalfTensor_validConv2Dptr,
    c_THHalfTensor_fullXCorr2Dptr,
    c_THHalfTensor_fullConv2Dptr,
    c_THHalfTensor_validXCorr2DRevptr,
    c_THHalfTensor_conv2DRevger,
    c_THHalfTensor_conv2DRevgerm,
    c_THHalfTensor_conv2Dger,
    c_THHalfTensor_conv2Dmv,
    c_THHalfTensor_conv2Dmm,
    c_THHalfTensor_conv2Dmul,
    c_THHalfTensor_conv2Dcmul,
    c_THHalfTensor_validXCorr3Dptr,
    c_THHalfTensor_validConv3Dptr,
    c_THHalfTensor_fullXCorr3Dptr,
    c_THHalfTensor_fullConv3Dptr,
    c_THHalfTensor_validXCorr3DRevptr,
    c_THHalfTensor_conv3DRevger,
    c_THHalfTensor_conv3Dger,
    c_THHalfTensor_conv3Dmv,
    c_THHalfTensor_conv3Dmul,
    c_THHalfTensor_conv3Dcmul,
    p_THHalfTensor_validXCorr2Dptr,
    p_THHalfTensor_validConv2Dptr,
    p_THHalfTensor_fullXCorr2Dptr,
    p_THHalfTensor_fullConv2Dptr,
    p_THHalfTensor_validXCorr2DRevptr,
    p_THHalfTensor_conv2DRevger,
    p_THHalfTensor_conv2DRevgerm,
    p_THHalfTensor_conv2Dger,
    p_THHalfTensor_conv2Dmv,
    p_THHalfTensor_conv2Dmm,
    p_THHalfTensor_conv2Dmul,
    p_THHalfTensor_conv2Dcmul,
    p_THHalfTensor_validXCorr3Dptr,
    p_THHalfTensor_validConv3Dptr,
    p_THHalfTensor_fullXCorr3Dptr,
    p_THHalfTensor_fullConv3Dptr,
    p_THHalfTensor_validXCorr3DRevptr,
    p_THHalfTensor_conv3DRevger,
    p_THHalfTensor_conv3Dger,
    p_THHalfTensor_conv3Dmv,
    p_THHalfTensor_conv3Dmul,
    p_THHalfTensor_conv3Dcmul) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensor_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_validXCorr2Dptr"
  c_THHalfTensor_validXCorr2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_validConv2Dptr"
  c_THHalfTensor_validConv2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_fullXCorr2Dptr"
  c_THHalfTensor_fullXCorr2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_fullConv2Dptr"
  c_THHalfTensor_fullConv2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_validXCorr2DRevptr"
  c_THHalfTensor_validXCorr2DRevptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv2DRevger"
  c_THHalfTensor_conv2DRevger :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv2DRevgerm"
  c_THHalfTensor_conv2DRevgerm :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv2Dger"
  c_THHalfTensor_conv2Dger :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv2Dmv"
  c_THHalfTensor_conv2Dmv :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv2Dmm"
  c_THHalfTensor_conv2Dmm :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv2Dmul"
  c_THHalfTensor_conv2Dmul :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv2Dcmul"
  c_THHalfTensor_conv2Dcmul :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_validXCorr3Dptr"
  c_THHalfTensor_validXCorr3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_validConv3Dptr"
  c_THHalfTensor_validConv3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_fullXCorr3Dptr"
  c_THHalfTensor_fullXCorr3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_fullConv3Dptr"
  c_THHalfTensor_fullConv3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_validXCorr3DRevptr"
  c_THHalfTensor_validXCorr3DRevptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv3DRevger"
  c_THHalfTensor_conv3DRevger :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv3Dger"
  c_THHalfTensor_conv3Dger :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv3Dmv"
  c_THHalfTensor_conv3Dmv :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv3Dmul"
  c_THHalfTensor_conv3Dmul :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THHalfTensor_conv3Dcmul"
  c_THHalfTensor_conv3Dcmul :: (Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |p_THHalfTensor_validXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_validXCorr2Dptr"
  p_THHalfTensor_validXCorr2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_validConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_validConv2Dptr"
  p_THHalfTensor_validConv2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_fullXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_fullXCorr2Dptr"
  p_THHalfTensor_fullXCorr2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_fullConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_fullConv2Dptr"
  p_THHalfTensor_fullConv2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_validXCorr2DRevptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_validXCorr2DRevptr"
  p_THHalfTensor_validXCorr2DRevptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_conv2DRevger : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv2DRevger"
  p_THHalfTensor_conv2DRevger :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_conv2DRevgerm : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv2DRevgerm"
  p_THHalfTensor_conv2DRevgerm :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_conv2Dger : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv2Dger"
  p_THHalfTensor_conv2Dger :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_conv2Dmv : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv2Dmv"
  p_THHalfTensor_conv2Dmv :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_conv2Dmm : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv2Dmm"
  p_THHalfTensor_conv2Dmm :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_conv2Dmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv2Dmul"
  p_THHalfTensor_conv2Dmul :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_conv2Dcmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv2Dcmul"
  p_THHalfTensor_conv2Dcmul :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_validXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_validXCorr3Dptr"
  p_THHalfTensor_validXCorr3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_validConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_validConv3Dptr"
  p_THHalfTensor_validConv3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_fullXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_fullXCorr3Dptr"
  p_THHalfTensor_fullXCorr3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_fullConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_fullConv3Dptr"
  p_THHalfTensor_fullConv3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_validXCorr3DRevptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_validXCorr3DRevptr"
  p_THHalfTensor_validXCorr3DRevptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLong -> CLong -> CLong -> Ptr THHalf -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_conv3DRevger : Pointer to r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv3DRevger"
  p_THHalfTensor_conv3DRevger :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THHalfTensor_conv3Dger : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv3Dger"
  p_THHalfTensor_conv3Dger :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_conv3Dmv : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv3Dmv"
  p_THHalfTensor_conv3Dmv :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_conv3Dmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv3Dmul"
  p_THHalfTensor_conv3Dmul :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_conv3Dcmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THHalfTensor_conv3Dcmul"
  p_THHalfTensor_conv3Dcmul :: FunPtr ((Ptr CTHHalfTensor) -> THHalf -> THHalf -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())