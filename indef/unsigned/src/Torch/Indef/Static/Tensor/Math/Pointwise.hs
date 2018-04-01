module Torch.Indef.Static.Tensor.Math.Pointwise where

import qualified Torch.Class.Tensor.Math.Pointwise        as Dynamic
import qualified Torch.Class.Tensor.Math.Pointwise.Static as Class

import Torch.Dimensions
import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor.Math.Pointwise ()

instance Class.TensorMathPointwise Tensor where
  sign_ :: Dimensions d => Tensor d -> Tensor d -> IO ()
  sign_ r t = Dynamic.sign_ (asDynamic r) (asDynamic t)
  cross_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> DimVal -> IO ()
  cross_ ret a b d = Dynamic.cross_ (asDynamic ret) (asDynamic a) (asDynamic b) d
  clamp_ :: Dimensions d => Tensor d -> Tensor d -> HsReal -> HsReal -> IO ()
  clamp_ ret a mn mx = Dynamic.clamp_ (asDynamic ret) (asDynamic a) mn mx
  cadd_ :: Dimensions d => Tensor d -> Tensor d -> HsReal -> Tensor d -> IO ()
  cadd_ ret a v b = Dynamic.cadd_ (asDynamic ret) (asDynamic a) v (asDynamic b)
  csub_ :: Dimensions d => Tensor d -> Tensor d -> HsReal -> Tensor d -> IO ()
  csub_ ret a v b = Dynamic.csub_ (asDynamic ret) (asDynamic a) v (asDynamic b)
  cmul_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cmul_ ret a b = Dynamic.cmul_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cpow_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cpow_ ret a b = Dynamic.cpow_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cdiv_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cdiv_ ret a b = Dynamic.cdiv_ (asDynamic ret) (asDynamic a) (asDynamic b)
  clshift_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  clshift_ ret a b = Dynamic.clshift_ (asDynamic ret) (asDynamic a) (asDynamic b)
  crshift_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  crshift_ ret a b = Dynamic.crshift_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cfmod_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cfmod_ ret a b = Dynamic.cfmod_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cremainder_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cremainder_ ret a b = Dynamic.cremainder_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cmax_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cmax_ ret a b = Dynamic.cmax_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cmin_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cmin_ ret a b = Dynamic.cmin_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cmaxValue_ :: Dimensions d => Tensor d -> Tensor d -> HsReal -> IO ()
  cmaxValue_ ret a v = Dynamic.cmaxValue_ (asDynamic ret) (asDynamic a) v
  cminValue_ :: Dimensions d => Tensor d -> Tensor d -> HsReal -> IO ()
  cminValue_ ret a v = Dynamic.cminValue_ (asDynamic ret) (asDynamic a) v
  cbitand_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cbitand_ ret a b = Dynamic.cbitand_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cbitor_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cbitor_ ret a b = Dynamic.cbitor_ (asDynamic ret) (asDynamic a) (asDynamic b)
  cbitxor_ :: Dimensions d => Tensor d -> Tensor d -> Tensor d -> IO ()
  cbitxor_ ret a b = Dynamic.cbitxor_ (asDynamic ret) (asDynamic a) (asDynamic b)

  addcmul_ :: Dimensions d => Tensor d -> Tensor d -> HsReal -> Tensor d -> Tensor d -> IO ()
  addcmul_ ret a v b c = Dynamic.addcmul_ (asDynamic ret) (asDynamic a) v (asDynamic b) (asDynamic c)
  addcdiv_ :: Dimensions d => Tensor d -> Tensor d -> HsReal -> Tensor d -> Tensor d -> IO ()
  addcdiv_ ret a v b c = Dynamic.addcdiv_ (asDynamic ret) (asDynamic a) v (asDynamic b) (asDynamic c)
