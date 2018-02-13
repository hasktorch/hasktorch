{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Dynamic.Conv where

import Data.Coerce (coerce)
import Foreign
import Foreign.C.Types
import qualified TensorConv as Sig
import qualified Torch.Class.C.Tensor.Conv as Class
import qualified Foreign.Marshal.Array as FM

import Torch.Core.Types

apply2DRevConv
  :: (Ptr CTensor -> CReal -> CReal -> Ptr CTensor -> Ptr CTensor -> CLLong -> CLLong -> IO ())
  -> Tensor -> HsReal -> HsReal -> Tensor -> Tensor -> Int64 -> Int64 -> IO ()
apply2DRevConv fn res a0 a1 t0 t1 b c =
  _with3Tensors res t0 t1 $ \res' t0' t1' ->
    fn res' (hs2cReal a0) (hs2cReal a1) t0' t1' (CLLong b) (CLLong c)

apply3DRevConv
  :: (Ptr CTensor -> CReal -> CReal -> Ptr CTensor -> Ptr CTensor -> CLLong -> CLLong -> CLLong -> IO ())
  -> Tensor -> HsReal -> HsReal -> Tensor -> Tensor -> Int64 -> Int64 -> Int64 -> IO ()
apply3DRevConv fn res a0 a1 t0 t1 b c d =
  _with3Tensors res t0 t1 $ \res' t0' t1' ->
    fn res' (hs2cReal a0) (hs2cReal a1) t0' t1' (CLLong b) (CLLong c) (CLLong d)

apply2DConv
  :: (Ptr CTensor -> CReal -> CReal -> Ptr CTensor -> Ptr CTensor -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  -> Tensor -> HsReal -> HsReal -> Tensor -> Tensor -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
apply2DConv fn res a0 a1 t0 t1 b c is0 is1 = do
  cchars0 <- newArray (coerce is0 :: [CChar])
  cchars1 <- newArray (coerce is1 :: [CChar])
  _with3Tensors res t0 t1 $ \res' t0' t1' ->
    fn res' (hs2cReal a0) (hs2cReal a1) t0' t1' (CLLong b) (CLLong c) cchars0 cchars1

apply3DConv
  :: (Ptr CTensor -> CReal -> CReal -> Ptr CTensor -> Ptr CTensor -> CLLong -> CLLong -> CLLong -> Ptr CChar -> Ptr CChar -> IO ())
  -> Tensor -> HsReal -> HsReal -> Tensor -> Tensor -> Int64 -> Int64 -> Int64 -> [Int8] -> [Int8] -> IO ()
apply3DConv fn res a0 a1 t0 t1 b c d is0 is1 = do
  cchars0 <- newArray (coerce is0 :: [CChar])
  cchars1 <- newArray (coerce is1 :: [CChar])
  _with3Tensors res t0 t1 $ \res' t0' t1' ->
    fn res' (hs2cReal a0) (hs2cReal a1) t0' t1' (CLLong b) (CLLong c) (CLLong d) cchars0 cchars1

instance Class.TensorConv Tensor where
  conv2DRevger  = apply2DRevConv Sig.c_conv2DRevger
  conv2DRevgerm = apply2DRevConv Sig.c_conv2DRevgerm
  conv2Dger     = apply2DConv    Sig.c_conv2Dger
  conv2Dmv      = apply2DConv    Sig.c_conv2Dmv
  conv2Dmm      = apply2DConv    Sig.c_conv2Dmm
  conv2Dmul     = apply2DConv    Sig.c_conv2Dmul
  conv2Dcmul    = apply2DConv    Sig.c_conv2Dcmul
  conv3DRevger  = apply3DRevConv Sig.c_conv3DRevger
  conv3Dger     = apply3DConv    Sig.c_conv3Dger
  conv3Dmv      = apply3DConv    Sig.c_conv3Dmv
  conv3Dmul     = apply3DConv    Sig.c_conv3Dmul
  conv3Dcmul    = apply3DConv    Sig.c_conv3Dcmul

