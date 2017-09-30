{-# LANGUAGE ForeignFunctionInterface #-}

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
    c_THLongTensor_conv3Dcmul,
    p_THLongTensor_validXCorr2Dptr,
    p_THLongTensor_validConv2Dptr,
    p_THLongTensor_fullXCorr2Dptr,
    p_THLongTensor_fullConv2Dptr,
    p_THLongTensor_validXCorr2DRevptr,
    p_THLongTensor_conv2DRevger,
    p_THLongTensor_conv2DRevgerm,
    p_THLongTensor_conv2Dger,
    p_THLongTensor_conv2Dmv,
    p_THLongTensor_conv2Dmm,
    p_THLongTensor_conv2Dmul,
    p_THLongTensor_conv2Dcmul,
    p_THLongTensor_validXCorr3Dptr,
    p_THLongTensor_validConv3Dptr,
    p_THLongTensor_fullXCorr3Dptr,
    p_THLongTensor_fullConv3Dptr,
    p_THLongTensor_validXCorr3DRevptr,
    p_THLongTensor_conv3DRevger,
    p_THLongTensor_conv3Dger,
    p_THLongTensor_conv3Dmv,
    p_THLongTensor_conv3Dmul,
    p_THLongTensor_conv3Dcmul) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_validXCorr2Dptr"
  c_THLongTensor_validXCorr2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_validConv2Dptr"
  c_THLongTensor_validConv2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_fullXCorr2Dptr"
  c_THLongTensor_fullXCorr2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_fullConv2Dptr"
  c_THLongTensor_fullConv2Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_validXCorr2DRevptr"
  c_THLongTensor_validXCorr2DRevptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv2DRevger"
  c_THLongTensor_conv2DRevger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv2DRevgerm"
  c_THLongTensor_conv2DRevgerm :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv2Dger"
  c_THLongTensor_conv2Dger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv2Dmv"
  c_THLongTensor_conv2Dmv :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv2Dmm"
  c_THLongTensor_conv2Dmm :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv2Dmul"
  c_THLongTensor_conv2Dmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv2Dcmul"
  c_THLongTensor_conv2Dcmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_validXCorr3Dptr"
  c_THLongTensor_validXCorr3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_validConv3Dptr"
  c_THLongTensor_validConv3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_fullXCorr3Dptr"
  c_THLongTensor_fullXCorr3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_fullConv3Dptr"
  c_THLongTensor_fullConv3Dptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_validXCorr3DRevptr"
  c_THLongTensor_validXCorr3DRevptr :: Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv3DRevger"
  c_THLongTensor_conv3DRevger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ()

-- |c_THLongTensor_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv3Dger"
  c_THLongTensor_conv3Dger :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv3Dmv"
  c_THLongTensor_conv3Dmv :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv3Dmul"
  c_THLongTensor_conv3Dmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h THLongTensor_conv3Dcmul"
  c_THLongTensor_conv3Dcmul :: (Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |p_THLongTensor_validXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_validXCorr2Dptr"
  p_THLongTensor_validXCorr2Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_validConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_validConv2Dptr"
  p_THLongTensor_validConv2Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_fullXCorr2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_fullXCorr2Dptr"
  p_THLongTensor_fullXCorr2Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_fullConv2Dptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_fullConv2Dptr"
  p_THLongTensor_fullConv2Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_validXCorr2DRevptr : Pointer to r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_validXCorr2DRevptr"
  p_THLongTensor_validXCorr2DRevptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_conv2DRevger : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv2DRevger"
  p_THLongTensor_conv2DRevger :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ())

-- |p_THLongTensor_conv2DRevgerm : Pointer to r_ beta alpha t_ k_ srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv2DRevgerm"
  p_THLongTensor_conv2DRevgerm :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> IO ())

-- |p_THLongTensor_conv2Dger : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv2Dger"
  p_THLongTensor_conv2Dger :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_conv2Dmv : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv2Dmv"
  p_THLongTensor_conv2Dmv :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_conv2Dmm : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv2Dmm"
  p_THLongTensor_conv2Dmm :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_conv2Dmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv2Dmul"
  p_THLongTensor_conv2Dmul :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_conv2Dcmul : Pointer to r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv2Dcmul"
  p_THLongTensor_conv2Dcmul :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_validXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_validXCorr3Dptr"
  p_THLongTensor_validXCorr3Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_validConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_validConv3Dptr"
  p_THLongTensor_validConv3Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_fullXCorr3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_fullXCorr3Dptr"
  p_THLongTensor_fullXCorr3Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_fullConv3Dptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_fullConv3Dptr"
  p_THLongTensor_fullConv3Dptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_validXCorr3DRevptr : Pointer to r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_validXCorr3DRevptr"
  p_THLongTensor_validXCorr3DRevptr :: FunPtr (Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_conv3DRevger : Pointer to r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv3DRevger"
  p_THLongTensor_conv3DRevger :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> IO ())

-- |p_THLongTensor_conv3Dger : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv3Dger"
  p_THLongTensor_conv3Dger :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_conv3Dmv : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv3Dmv"
  p_THLongTensor_conv3Dmv :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_conv3Dmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv3Dmul"
  p_THLongTensor_conv3Dmul :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_conv3Dcmul : Pointer to r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall unsafe "THTensorConv.h &THLongTensor_conv3Dcmul"
  p_THLongTensor_conv3Dcmul :: FunPtr ((Ptr CTHLongTensor) -> CLong -> CLong -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> CLong -> CLong -> CLong -> Ptr CChar -> Ptr CChar -> IO ())