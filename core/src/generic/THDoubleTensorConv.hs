{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleTensorConv (
    c_THDoubleTensor_validXCorr2Dptr,
    c_THDoubleTensor_validConv2Dptr,
    c_THDoubleTensor_fullXCorr2Dptr,
    c_THDoubleTensor_fullConv2Dptr,
    c_THDoubleTensor_validXCorr2DRevptr,
    c_THDoubleTensor_conv2DRevger,
    c_THDoubleTensor_conv2DRevgerm,
    c_THDoubleTensor_conv2Dger,
    c_THDoubleTensor_conv2Dmv,
    c_THDoubleTensor_conv2Dmm,
    c_THDoubleTensor_conv2Dmul,
    c_THDoubleTensor_conv2Dcmul,
    c_THDoubleTensor_validXCorr3Dptr,
    c_THDoubleTensor_validConv3Dptr,
    c_THDoubleTensor_fullXCorr3Dptr,
    c_THDoubleTensor_fullConv3Dptr,
    c_THDoubleTensor_validXCorr3DRevptr,
    c_THDoubleTensor_conv3DRevger,
    c_THDoubleTensor_conv3Dger,
    c_THDoubleTensor_conv3Dmv,
    c_THDoubleTensor_conv3Dmul,
    c_THDoubleTensor_conv3Dcmul,
    p_THDoubleTensor_validXCorr2Dptr,
    p_THDoubleTensor_validConv2Dptr,
    p_THDoubleTensor_fullXCorr2Dptr,
    p_THDoubleTensor_fullConv2Dptr,
    p_THDoubleTensor_validXCorr2DRevptr,
    p_THDoubleTensor_conv2DRevger,
    p_THDoubleTensor_conv2DRevgerm,
    p_THDoubleTensor_conv2Dger,
    p_THDoubleTensor_conv2Dmv,
    p_THDoubleTensor_conv2Dmm,
    p_THDoubleTensor_conv2Dmul,
    p_THDoubleTensor_conv2Dcmul,
    p_THDoubleTensor_validXCorr3Dptr,
    p_THDoubleTensor_validConv3Dptr,
    p_THDoubleTensor_fullXCorr3Dptr,
    p_THDoubleTensor_fullConv3Dptr,
    p_THDoubleTensor_validXCorr3DRevptr,
    p_THDoubleTensor_conv3DRevger,
    p_THDoubleTensor_conv3Dger,
    p_THDoubleTensor_conv3Dmv,
    p_THDoubleTensor_conv3Dmul,
    p_THDoubleTensor_conv3Dcmul) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensor_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_validXCorr2Dptr"
  c_THDoubleTensor_validXCorr2Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_validConv2Dptr"
  c_THDoubleTensor_validConv2Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_fullXCorr2Dptr"
  c_THDoubleTensor_fullXCorr2Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_fullConv2Dptr"
  c_THDoubleTensor_fullConv2Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_validXCorr2DRevptr"
  c_THDoubleTensor_validXCorr2DRevptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv2DRevger"
  c_THDoubleTensor_conv2DRevger :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv2DRevgerm"
  c_THDoubleTensor_conv2DRevgerm :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv2Dger"
  c_THDoubleTensor_conv2Dger :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv2Dmv"
  c_THDoubleTensor_conv2Dmv :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv2Dmm"
  c_THDoubleTensor_conv2Dmm :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv2Dmul"
  c_THDoubleTensor_conv2Dmul :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv2Dcmul"
  c_THDoubleTensor_conv2Dcmul :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_validXCorr3Dptr"
  c_THDoubleTensor_validXCorr3Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_validConv3Dptr"
  c_THDoubleTensor_validConv3Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_fullXCorr3Dptr"
  c_THDoubleTensor_fullXCorr3Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_fullConv3Dptr"
  c_THDoubleTensor_fullConv3Dptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_validXCorr3DRevptr"
  c_THDoubleTensor_validXCorr3DRevptr :: Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv3DRevger"
  c_THDoubleTensor_conv3DRevger :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv3Dger"
  c_THDoubleTensor_conv3Dger :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv3Dmv"
  c_THDoubleTensor_conv3Dmv :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv3Dmul"
  c_THDoubleTensor_conv3Dmul :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THDoubleTensor_conv3Dcmul"
  c_THDoubleTensor_conv3Dcmul :: (Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |p_THDoubleTensor_validXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_validXCorr2Dptr"
  p_THDoubleTensor_validXCorr2Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_validConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_validConv2Dptr"
  p_THDoubleTensor_validConv2Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_fullXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_fullXCorr2Dptr"
  p_THDoubleTensor_fullXCorr2Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_fullConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_fullConv2Dptr"
  p_THDoubleTensor_fullConv2Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_validXCorr2DRevptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_validXCorr2DRevptr"
  p_THDoubleTensor_validXCorr2DRevptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_conv2DRevger : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv2DRevger"
  p_THDoubleTensor_conv2DRevger :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_conv2DRevgerm : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv2DRevgerm"
  p_THDoubleTensor_conv2DRevgerm :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_conv2Dger : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv2Dger"
  p_THDoubleTensor_conv2Dger :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_conv2Dmv : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv2Dmv"
  p_THDoubleTensor_conv2Dmv :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_conv2Dmm : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv2Dmm"
  p_THDoubleTensor_conv2Dmm :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_conv2Dmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv2Dmul"
  p_THDoubleTensor_conv2Dmul :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_conv2Dcmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv2Dcmul"
  p_THDoubleTensor_conv2Dcmul :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_validXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_validXCorr3Dptr"
  p_THDoubleTensor_validXCorr3Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_validConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_validConv3Dptr"
  p_THDoubleTensor_validConv3Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_fullXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_fullXCorr3Dptr"
  p_THDoubleTensor_fullXCorr3Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_fullConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_fullConv3Dptr"
  p_THDoubleTensor_fullConv3Dptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_validXCorr3DRevptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_validXCorr3DRevptr"
  p_THDoubleTensor_validXCorr3DRevptr :: FunPtr (Ptr CDouble -> CDouble -> Ptr CDouble -> CLong -> CLong -> CLong -> Ptr CDouble -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_conv3DRevger : Pointer to r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv3DRevger"
  p_THDoubleTensor_conv3DRevger :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_conv3Dger : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv3Dger"
  p_THDoubleTensor_conv3Dger :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_conv3Dmv : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv3Dmv"
  p_THDoubleTensor_conv3Dmv :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_conv3Dmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv3Dmul"
  p_THDoubleTensor_conv3Dmul :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_conv3Dcmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THDoubleTensor_conv3Dcmul"
  p_THDoubleTensor_conv3Dcmul :: FunPtr ((Ptr CTHDoubleTensor) -> CDouble -> CDouble -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())