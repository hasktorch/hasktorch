{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatTensorConv (
    c_THFloatTensor_validXCorr2Dptr,
    c_THFloatTensor_validConv2Dptr,
    c_THFloatTensor_fullXCorr2Dptr,
    c_THFloatTensor_fullConv2Dptr,
    c_THFloatTensor_validXCorr2DRevptr,
    c_THFloatTensor_conv2DRevger,
    c_THFloatTensor_conv2DRevgerm,
    c_THFloatTensor_conv2Dger,
    c_THFloatTensor_conv2Dmv,
    c_THFloatTensor_conv2Dmm,
    c_THFloatTensor_conv2Dmul,
    c_THFloatTensor_conv2Dcmul,
    c_THFloatTensor_validXCorr3Dptr,
    c_THFloatTensor_validConv3Dptr,
    c_THFloatTensor_fullXCorr3Dptr,
    c_THFloatTensor_fullConv3Dptr,
    c_THFloatTensor_validXCorr3DRevptr,
    c_THFloatTensor_conv3DRevger,
    c_THFloatTensor_conv3Dger,
    c_THFloatTensor_conv3Dmv,
    c_THFloatTensor_conv3Dmul,
    c_THFloatTensor_conv3Dcmul,
    p_THFloatTensor_validXCorr2Dptr,
    p_THFloatTensor_validConv2Dptr,
    p_THFloatTensor_fullXCorr2Dptr,
    p_THFloatTensor_fullConv2Dptr,
    p_THFloatTensor_validXCorr2DRevptr,
    p_THFloatTensor_conv2DRevger,
    p_THFloatTensor_conv2DRevgerm,
    p_THFloatTensor_conv2Dger,
    p_THFloatTensor_conv2Dmv,
    p_THFloatTensor_conv2Dmm,
    p_THFloatTensor_conv2Dmul,
    p_THFloatTensor_conv2Dcmul,
    p_THFloatTensor_validXCorr3Dptr,
    p_THFloatTensor_validConv3Dptr,
    p_THFloatTensor_fullXCorr3Dptr,
    p_THFloatTensor_fullConv3Dptr,
    p_THFloatTensor_validXCorr3DRevptr,
    p_THFloatTensor_conv3DRevger,
    p_THFloatTensor_conv3Dger,
    p_THFloatTensor_conv3Dmv,
    p_THFloatTensor_conv3Dmul,
    p_THFloatTensor_conv3Dcmul) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatTensor_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_validXCorr2Dptr"
  c_THFloatTensor_validXCorr2Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_validConv2Dptr"
  c_THFloatTensor_validConv2Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_fullXCorr2Dptr"
  c_THFloatTensor_fullXCorr2Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_fullConv2Dptr"
  c_THFloatTensor_fullConv2Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_validXCorr2DRevptr"
  c_THFloatTensor_validXCorr2DRevptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv2DRevger"
  c_THFloatTensor_conv2DRevger :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv2DRevgerm"
  c_THFloatTensor_conv2DRevgerm :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv2Dger"
  c_THFloatTensor_conv2Dger :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv2Dmv"
  c_THFloatTensor_conv2Dmv :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv2Dmm"
  c_THFloatTensor_conv2Dmm :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv2Dmul"
  c_THFloatTensor_conv2Dmul :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv2Dcmul"
  c_THFloatTensor_conv2Dcmul :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_validXCorr3Dptr"
  c_THFloatTensor_validXCorr3Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_validConv3Dptr"
  c_THFloatTensor_validConv3Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_fullXCorr3Dptr"
  c_THFloatTensor_fullXCorr3Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_fullConv3Dptr"
  c_THFloatTensor_fullConv3Dptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_validXCorr3DRevptr"
  c_THFloatTensor_validXCorr3DRevptr :: Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv3DRevger"
  c_THFloatTensor_conv3DRevger :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv3Dger"
  c_THFloatTensor_conv3Dger :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv3Dmv"
  c_THFloatTensor_conv3Dmv :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv3Dmul"
  c_THFloatTensor_conv3Dmul :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THFloatTensor_conv3Dcmul"
  c_THFloatTensor_conv3Dcmul :: (Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |p_THFloatTensor_validXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_validXCorr2Dptr"
  p_THFloatTensor_validXCorr2Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_validConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_validConv2Dptr"
  p_THFloatTensor_validConv2Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_fullXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_fullXCorr2Dptr"
  p_THFloatTensor_fullXCorr2Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_fullConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_fullConv2Dptr"
  p_THFloatTensor_fullConv2Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_validXCorr2DRevptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_validXCorr2DRevptr"
  p_THFloatTensor_validXCorr2DRevptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_conv2DRevger : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv2DRevger"
  p_THFloatTensor_conv2DRevger :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_conv2DRevgerm : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv2DRevgerm"
  p_THFloatTensor_conv2DRevgerm :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_conv2Dger : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv2Dger"
  p_THFloatTensor_conv2Dger :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_conv2Dmv : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv2Dmv"
  p_THFloatTensor_conv2Dmv :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_conv2Dmm : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv2Dmm"
  p_THFloatTensor_conv2Dmm :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_conv2Dmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv2Dmul"
  p_THFloatTensor_conv2Dmul :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_conv2Dcmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv2Dcmul"
  p_THFloatTensor_conv2Dcmul :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_validXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_validXCorr3Dptr"
  p_THFloatTensor_validXCorr3Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_validConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_validConv3Dptr"
  p_THFloatTensor_validConv3Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_fullXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_fullXCorr3Dptr"
  p_THFloatTensor_fullXCorr3Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_fullConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_fullConv3Dptr"
  p_THFloatTensor_fullConv3Dptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_validXCorr3DRevptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_validXCorr3DRevptr"
  p_THFloatTensor_validXCorr3DRevptr :: FunPtr (Ptr CFloat -> CFloat -> Ptr CFloat -> CLong -> CLong -> CLong -> Ptr CFloat -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_conv3DRevger : Pointer to r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv3DRevger"
  p_THFloatTensor_conv3DRevger :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THFloatTensor_conv3Dger : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv3Dger"
  p_THFloatTensor_conv3Dger :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_conv3Dmv : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv3Dmv"
  p_THFloatTensor_conv3Dmv :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_conv3Dmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv3Dmul"
  p_THFloatTensor_conv3Dmul :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_conv3Dcmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THFloatTensor_conv3Dcmul"
  p_THFloatTensor_conv3Dcmul :: FunPtr ((Ptr CTHFloatTensor) -> CFloat -> CFloat -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())