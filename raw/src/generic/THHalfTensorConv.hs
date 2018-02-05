{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfTensorConv
  ( c_validXCorr2Dptr
  , c_validConv2Dptr
  , c_fullXCorr2Dptr
  , c_fullConv2Dptr
  , c_validXCorr2DRevptr
  , c_conv2DRevger
  , c_conv2DRevgerm
  , c_conv2Dger
  , c_conv2Dmv
  , c_conv2Dmm
  , c_conv2Dmul
  , c_conv2Dcmul
  , c_validXCorr3Dptr
  , c_validConv3Dptr
  , c_fullXCorr3Dptr
  , c_fullConv3Dptr
  , c_validXCorr3DRevptr
  , c_conv3DRevger
  , c_conv3Dger
  , c_conv3Dmv
  , c_conv3Dmul
  , c_conv3Dcmul
  , p_validXCorr2Dptr
  , p_validConv2Dptr
  , p_fullXCorr2Dptr
  , p_fullConv2Dptr
  , p_validXCorr2DRevptr
  , p_conv2DRevger
  , p_conv2DRevgerm
  , p_conv2Dger
  , p_conv2Dmv
  , p_conv2Dmm
  , p_conv2Dmul
  , p_conv2Dcmul
  , p_validXCorr3Dptr
  , p_validConv3Dptr
  , p_fullXCorr3Dptr
  , p_fullConv3Dptr
  , p_validXCorr3DRevptr
  , p_conv3DRevger
  , p_conv3Dger
  , p_conv3Dmv
  , p_conv3Dmul
  , p_conv3Dcmul
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_validXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h validXCorr2Dptr"
  c_validXCorr2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_validConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h validConv2Dptr"
  c_validConv2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_fullXCorr2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h fullXCorr2Dptr"
  c_fullXCorr2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_fullConv2Dptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h fullConv2Dptr"
  c_fullConv2Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_validXCorr2DRevptr : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h validXCorr2DRevptr"
  c_validXCorr2DRevptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_conv2DRevger : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h conv2DRevger"
  c_conv2DRevger :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> IO ()

-- | c_conv2DRevgerm : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h conv2DRevgerm"
  c_conv2DRevgerm :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> IO ()

-- | c_conv2Dger : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv2Dger"
  c_conv2Dger :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_conv2Dmv : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv2Dmv"
  c_conv2Dmv :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_conv2Dmm : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv2Dmm"
  c_conv2Dmm :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_conv2Dmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv2Dmul"
  c_conv2Dmul :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_conv2Dcmul : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv2Dcmul"
  c_conv2Dcmul :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_validXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h validXCorr3Dptr"
  c_validXCorr3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_validConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h validConv3Dptr"
  c_validConv3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_fullXCorr3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h fullXCorr3Dptr"
  c_fullXCorr3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_fullConv3Dptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h fullConv3Dptr"
  c_fullConv3Dptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_validXCorr3DRevptr : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h validXCorr3DRevptr"
  c_validXCorr3DRevptr :: Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_conv3DRevger : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall "THTensorConv.h conv3DRevger"
  c_conv3DRevger :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_conv3Dger : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv3Dger"
  c_conv3Dger :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_conv3Dmv : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv3Dmv"
  c_conv3Dmv :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_conv3Dmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv3Dmul"
  c_conv3Dmul :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_conv3Dcmul : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h conv3Dcmul"
  c_conv3Dcmul :: Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ()

-- |p_validXCorr2Dptr : Pointer to function : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h &validXCorr2Dptr"
  p_validXCorr2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_validConv2Dptr : Pointer to function : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h &validConv2Dptr"
  p_validConv2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_fullXCorr2Dptr : Pointer to function : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h &fullXCorr2Dptr"
  p_fullXCorr2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_fullConv2Dptr : Pointer to function : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h &fullConv2Dptr"
  p_fullConv2Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_validXCorr2DRevptr : Pointer to function : r_ alpha t_ ir ic k_ kr kc sr sc -> void
foreign import ccall "THTensorConv.h &validXCorr2DRevptr"
  p_validXCorr2DRevptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_conv2DRevger : Pointer to function : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h &conv2DRevger"
  p_conv2DRevger :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> IO ())

-- |p_conv2DRevgerm : Pointer to function : r_ beta alpha t_ k_ srow scol -> void
foreign import ccall "THTensorConv.h &conv2DRevgerm"
  p_conv2DRevgerm :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> IO ())

-- |p_conv2Dger : Pointer to function : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv2Dger"
  p_conv2Dger :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_conv2Dmv : Pointer to function : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv2Dmv"
  p_conv2Dmv :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_conv2Dmm : Pointer to function : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv2Dmm"
  p_conv2Dmm :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_conv2Dmul : Pointer to function : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv2Dmul"
  p_conv2Dmul :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_conv2Dcmul : Pointer to function : r_ beta alpha t_ k_ srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv2Dcmul"
  p_conv2Dcmul :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_validXCorr3Dptr : Pointer to function : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h &validXCorr3Dptr"
  p_validXCorr3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_validConv3Dptr : Pointer to function : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h &validConv3Dptr"
  p_validConv3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_fullXCorr3Dptr : Pointer to function : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h &fullXCorr3Dptr"
  p_fullXCorr3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_fullConv3Dptr : Pointer to function : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h &fullConv3Dptr"
  p_fullConv3Dptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_validXCorr3DRevptr : Pointer to function : r_ alpha t_ it ir ic k_ kt kr kc st sr sc -> void
foreign import ccall "THTensorConv.h &validXCorr3DRevptr"
  p_validXCorr3DRevptr :: FunPtr (Ptr THHalf -> THHalf -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> Ptr THHalf -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_conv3DRevger : Pointer to function : r_ beta alpha t_ k_ sdepth srow scol -> void
foreign import ccall "THTensorConv.h &conv3DRevger"
  p_conv3DRevger :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> IO ())

-- |p_conv3Dger : Pointer to function : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv3Dger"
  p_conv3Dger :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_conv3Dmv : Pointer to function : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv3Dmv"
  p_conv3Dmv :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_conv3Dmul : Pointer to function : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv3Dmul"
  p_conv3Dmul :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_conv3Dcmul : Pointer to function : r_ beta alpha t_ k_ sdepth srow scol vf xc -> void
foreign import ccall "THTensorConv.h &conv3Dcmul"
  p_conv3Dcmul :: FunPtr (Ptr CTHHalfTensor -> THHalf -> THHalf -> Ptr CTHHalfTensor -> Ptr CTHHalfTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())