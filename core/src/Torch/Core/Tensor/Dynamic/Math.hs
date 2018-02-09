module Torch.Core.Tensor.Dynamic.Math
  ( TensorMath(..)
  ) where

import Torch.Class.C.Internal
import GHC.Int
import qualified Torch.Class.C.Tensor.Math as CCall

import THTypes
import Foreign
import qualified Torch.Core.LongTensor.Dynamic   as L
import qualified Torch.Core.FloatTensor.Dynamic  as F
import qualified Torch.Core.ByteTensor.Dynamic   as B
-- import qualified Torch.Core.CharTensor.Dynamic   as C
import qualified Torch.Core.ShortTensor.Dynamic  as S
import qualified Torch.Core.IntTensor.Dynamic    as I
import qualified Torch.Core.DoubleTensor.Dynamic as D
-- import qualified Torch.Core.HalfTensor.Dynamic   as H
--
import qualified Torch.Core.LongStorage as LS

type ByteTensor = B.Tensor
type LongTensor = L.Tensor
type LongStorage = LS.Storage

class CCall.TensorMath t => TensorMath t where
  fill :: t -> HsReal t -> IO ()
  fill = CCall.fill

  zero :: t -> IO ()
  zero = CCall.zero

  maskedFill :: t -> ByteTensor -> HsReal t -> IO ()
  maskedFill t b s = withForeignPtr (B.tensor b) (\b' -> CCall.maskedFill t b' s)

  maskedCopy :: t -> ByteTensor -> t -> IO ()
  maskedCopy t0 b t1 = withForeignPtr (B.tensor b) (\b' -> CCall.maskedCopy t0 b' t1)

  maskedSelect :: t -> t -> ByteTensor -> IO ()
  maskedSelect t0 t1 b = withForeignPtr (B.tensor b) (CCall.maskedSelect t0 t1)

  nonzero :: LongTensor -> t -> IO ()
  nonzero l t = withForeignPtr (L.tensor l) (`CCall.nonzero` t)

  indexSelect :: t -> t -> Int32 -> LongTensor -> IO ()
  indexSelect t0 t1 i l = withForeignPtr (L.tensor l) (\l' -> CCall.indexSelect t0 t1 i l')

  indexCopy :: t -> Int32 -> LongTensor -> t -> IO ()
  indexCopy t0 i l t1 = withForeignPtr (L.tensor l) (\l' -> CCall.indexCopy t0 i l' t1)

  indexAdd :: t -> Int32 -> LongTensor -> t -> IO ()
  indexAdd t0 i l t1 = withForeignPtr (L.tensor l) (\l' -> CCall.indexAdd t0 i l' t1)

  indexFill :: t -> Int32 -> LongTensor -> HsReal t -> IO ()
  indexFill t i l v = withForeignPtr (L.tensor l) (\l' -> CCall.indexFill t i l' v)

  take :: t -> t -> LongTensor -> IO ()
  take t0 t1 l = withForeignPtr (L.tensor l) (\l' -> CCall.take t0 t1 l')

  put :: t -> LongTensor -> t -> Int32 -> IO ()
  put t0 l t1 i = withForeignPtr (L.tensor l) (\l' -> CCall.put t0 l' t1 i)

  gather :: t -> t -> Int32 -> LongTensor -> IO ()
  gather t0 t1 i l = withForeignPtr (L.tensor l) (\l' -> CCall.gather t0 t1 i l')

  scatter :: t -> Int32 -> LongTensor -> t -> IO ()
  scatter t0 i l t1 = withForeignPtr (L.tensor l) (\l' -> CCall.scatter t0 i l' t1)

  scatterAdd :: t -> Int32 -> LongTensor -> t -> IO ()
  scatterAdd t0 i l t1 = withForeignPtr (L.tensor l) (\l' -> CCall.scatterAdd t0 i l' t1)

  scatterFill  :: t -> Int32 -> LongTensor -> HsReal t -> IO ()
  scatterFill t0 i l v = withForeignPtr (L.tensor l) (\l' -> CCall.scatterFill t0 i l' v)

  dot          :: t -> t -> IO (HsAccReal t)
  dot          = CCall.dot
  minall       :: t -> IO (HsReal t)
  minall       = CCall.minall
  maxall       :: t -> IO (HsReal t)
  maxall       = CCall.maxall
  medianall    :: t -> IO (HsReal t)
  medianall    = CCall.medianall
  sumall       :: t -> IO (HsAccReal t)
  sumall       = CCall.sumall
  prodall      :: t -> IO (HsAccReal t)
  prodall      = CCall.prodall
  add          :: t -> t -> HsReal t -> IO ()
  add          = CCall.add
  sub          :: t -> t -> HsReal t -> IO ()
  sub          = CCall.sub
  add_scaled   :: t -> t -> HsReal t -> HsReal t -> IO ()
  add_scaled   = CCall.add_scaled
  sub_scaled   :: t -> t -> HsReal t -> HsReal t -> IO ()
  sub_scaled   = CCall.sub_scaled
  mul          :: t -> t -> HsReal t -> IO ()
  mul          = CCall.mul
  div          :: t -> t -> HsReal t -> IO ()
  div          = CCall.div
  lshift       :: t -> t -> HsReal t -> IO ()
  lshift       = CCall.lshift
  rshift       :: t -> t -> HsReal t -> IO ()
  rshift       = CCall.rshift
  fmod         :: t -> t -> HsReal t -> IO ()
  fmod         = CCall.fmod
  remainder    :: t -> t -> HsReal t -> IO ()
  remainder    = CCall.remainder
  clamp        :: t -> t -> HsReal t -> HsReal t -> IO ()
  clamp        = CCall.clamp
  bitand       :: t -> t -> HsReal t -> IO ()
  bitand       = CCall.bitand
  bitor        :: t -> t -> HsReal t -> IO ()
  bitor        = CCall.bitor
  bitxor       :: t -> t -> HsReal t -> IO ()
  bitxor       = CCall.bitxor
  cadd         :: t -> t -> HsReal t -> t -> IO ()
  cadd         = CCall.cadd
  csub         :: t -> t -> HsReal t -> t -> IO ()
  csub         = CCall.csub
  cmul         :: t -> t -> t -> IO ()
  cmul         = CCall.cmul
  cpow         :: t -> t -> t -> IO ()
  cpow         = CCall.cpow
  cdiv         :: t -> t -> t -> IO ()
  cdiv         = CCall.cdiv
  clshift      :: t -> t -> t -> IO ()
  clshift      = CCall.clshift
  crshift      :: t -> t -> t -> IO ()
  crshift      = CCall.crshift
  cfmod        :: t -> t -> t -> IO ()
  cfmod        = CCall.cfmod
  cremainder   :: t -> t -> t -> IO ()
  cremainder   = CCall.cremainder
  cbitand      :: t -> t -> t -> IO ()
  cbitand      = CCall.cbitand
  cbitor       :: t -> t -> t -> IO ()
  cbitor       = CCall.cbitor
  cbitxor      :: t -> t -> t -> IO ()
  cbitxor      = CCall.cbitxor
  addcmul      :: t -> t -> HsReal t -> t -> t -> IO ()
  addcmul      = CCall.addcmul
  addcdiv      :: t -> t -> HsReal t -> t -> t -> IO ()
  addcdiv      = CCall.addcdiv
  addmv        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addmv        = CCall.addmv
  addmm        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addmm        = CCall.addmm
  addr         :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addr         = CCall.addr
  addbmm       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addbmm       = CCall.addbmm
  baddbmm      :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  baddbmm      = CCall.baddbmm
  match        :: t -> t -> t -> HsReal t -> IO ()
  match        = CCall.match
  numel        :: t -> IO Int64
  numel        = CCall.numel




  max :: t -> LongTensor -> t -> Int32 -> Int32 -> IO ()
  max t0 l t1 i0 i1 = withForeignPtr (L.tensor l) $ \l' -> CCall.max t0 l' t1 i0 i1

  min :: t -> LongTensor -> t -> Int32 -> Int32 -> IO ()
  min t0 l t1 i0 i1 = withForeignPtr (L.tensor l) $ \l' -> CCall.min t0 l' t1 i0 i1

  kthvalue :: t -> LongTensor -> t -> Int64 -> Int32 -> Int32 -> IO ()
  kthvalue t0 l t1 i0 i1 i2 = withForeignPtr (L.tensor l) $ \l' -> CCall.kthvalue t0 l' t1 i0 i1 i2

  mode :: t -> LongTensor -> t -> Int32 -> Int32 -> IO ()
  mode t0 l t1 i0 i1 = withForeignPtr (L.tensor l) $ \l' -> CCall.mode t0 l' t1 i0 i1

  median :: t -> LongTensor -> t -> Int32 -> Int32 -> IO ()
  median t0 l t1 i0 i1 = withForeignPtr (L.tensor l) $ \l' -> CCall.median t0 l' t1 i0 i1

  sum          :: t -> t -> Int32 -> Int32 -> IO ()
  sum          = CCall.sum
  prod         :: t -> t -> Int32 -> Int32 -> IO ()
  prod         = CCall.prod
  cumsum       :: t -> t -> Int32 -> IO ()
  cumsum       = CCall.cumsum
  cumprod      :: t -> t -> Int32 -> IO ()
  cumprod      = CCall.cumprod
  sign         :: t -> t -> IO ()
  sign         = CCall.sign
  trace        :: t -> IO (HsAccReal t)
  trace        = CCall.trace
  cross        :: t -> t -> t -> Int32 -> IO ()
  cross        = CCall.cross
  cmax         :: t -> t -> t -> IO ()
  cmax         = CCall.cmax
  cmin         :: t -> t -> t -> IO ()
  cmin         = CCall.cmin
  cmaxValue    :: t -> t -> HsReal t -> IO ()
  cmaxValue    = CCall.cmaxValue
  cminValue    :: t -> t -> HsReal t -> IO ()
  cminValue    = CCall.cminValue

  zeros :: t -> LongStorage -> IO ()
  zeros t l = withForeignPtr (LS.storage l) $ \l' -> CCall.zeros t l'

  zerosLike    :: t -> t -> IO ()
  zerosLike    = CCall.zerosLike

  ones :: t -> LongStorage -> IO ()
  ones t l = withForeignPtr (LS.storage l) $ \l' -> CCall.ones t l'

  onesLike     :: t -> t -> IO ()
  onesLike     = CCall.onesLike
  diag         :: t -> t -> Int32 -> IO ()
  diag         = CCall.diag
  eye          :: t -> Int64 -> Int64 -> IO ()
  eye          = CCall.eye
  arange       :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
  arange       = CCall.arange
  range        :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
  range        = CCall.range

  randperm     :: t -> Ptr CTHGenerator -> Int64 -> IO ()
  randperm     = CCall.randperm
  -- randperm t l i = withForeignPtr (L.tensor l) $ \l' -> CCall.randperm t l' i

  reshape :: t -> t -> LongStorage -> IO ()
  reshape t0 t1 l = withForeignPtr (LS.storage l) $ \l' -> CCall.reshape t0 t1 l'

  sort :: t -> LongTensor -> t -> Int32 -> Int32 -> IO ()
  sort t0 l t1 i0 i1 = withForeignPtr (L.tensor l) $ \l' -> CCall.sort t0 l' t1 i0 i1

  topk :: t -> LongTensor -> t -> Int64 -> Int32 -> Int32 -> Int32 -> IO ()
  topk t0 l t1 i0 i1 i2 i3 = withForeignPtr (L.tensor l) $ \l' -> CCall.topk t0 l' t1 i0 i1 i2 i3

  tril         :: t -> t -> Int64 -> IO ()
  tril         = CCall.tril
  triu         :: t -> t -> Int64 -> IO ()
  triu         = CCall.triu
  cat          :: t -> t -> t -> Int32 -> IO ()
  cat          = CCall.cat
  catArray     :: t -> [t] -> Int32 -> Int32 -> IO ()
  catArray     = CCall.catArray
  equal        :: t -> t -> IO Int32
  equal        = CCall.equal

  ltValue :: ByteTensor -> t -> HsReal t -> IO ()
  ltValue b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.ltValue b' t v
  leValue :: ByteTensor -> t -> HsReal t -> IO ()
  leValue b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.leValue b' t v
  gtValue :: ByteTensor -> t -> HsReal t -> IO ()
  gtValue b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.gtValue b' t v
  geValue :: ByteTensor -> t -> HsReal t -> IO ()
  geValue b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.geValue b' t v
  neValue :: ByteTensor -> t -> HsReal t -> IO ()
  neValue b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.neValue b' t v
  eqValue :: ByteTensor -> t -> HsReal t -> IO ()
  eqValue b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.eqValue b' t v

  ltValueT :: t -> t -> HsReal t -> IO ()
  ltValueT = CCall.ltValueT
  leValueT :: t -> t -> HsReal t -> IO ()
  leValueT = CCall.leValueT
  gtValueT :: t -> t -> HsReal t -> IO ()
  gtValueT = CCall.gtValueT
  geValueT :: t -> t -> HsReal t -> IO ()
  geValueT = CCall.geValueT
  neValueT :: t -> t -> HsReal t -> IO ()
  neValueT = CCall.neValueT
  eqValueT :: t -> t -> HsReal t -> IO ()
  eqValueT = CCall.eqValueT


  ltTensor :: ByteTensor -> t -> t -> IO ()
  ltTensor b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.ltTensor b' t v
  leTensor :: ByteTensor -> t -> t -> IO ()
  leTensor b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.leTensor b' t v
  gtTensor :: ByteTensor -> t -> t -> IO ()
  gtTensor b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.gtTensor b' t v
  geTensor :: ByteTensor -> t -> t -> IO ()
  geTensor b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.geTensor b' t v
  neTensor :: ByteTensor -> t -> t -> IO ()
  neTensor b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.neTensor b' t v
  eqTensor :: ByteTensor -> t -> t -> IO ()
  eqTensor b t v = withForeignPtr (B.tensor b) $ \b' -> CCall.eqTensor b' t v

  ltTensorT    :: t -> t -> t -> IO ()
  ltTensorT    = CCall.ltTensorT
  leTensorT    :: t -> t -> t -> IO ()
  leTensorT    = CCall.leTensorT
  gtTensorT    :: t -> t -> t -> IO ()
  gtTensorT    = CCall.gtTensorT
  geTensorT    :: t -> t -> t -> IO ()
  geTensorT    = CCall.geTensorT
  neTensorT    :: t -> t -> t -> IO ()
  neTensorT    = CCall.neTensorT
  eqTensorT    :: t -> t -> t -> IO ()
  eqTensorT    = CCall.eqTensorT

instance TensorMath B.Tensor where
instance TensorMath S.Tensor where
instance TensorMath I.Tensor where
instance TensorMath L.Tensor where
instance TensorMath F.Tensor where
instance TensorMath D.Tensor where

class TensorMathSigned t where
  neg          :: t -> t -> IO ()
  abs          :: t -> t -> IO ()

class TensorMathFloating t where
  cinv         :: t -> t -> IO ()
  sigmoid      :: t -> t -> IO ()
  log          :: t -> t -> IO ()
  lgamma       :: t -> t -> IO ()
  log1p        :: t -> t -> IO ()
  exp          :: t -> t -> IO ()
  cos          :: t -> t -> IO ()
  acos         :: t -> t -> IO ()
  cosh         :: t -> t -> IO ()
  sin          :: t -> t -> IO ()
  asin         :: t -> t -> IO ()
  sinh         :: t -> t -> IO ()
  tan          :: t -> t -> IO ()
  atan         :: t -> t -> IO ()
  atan2        :: t -> t -> t -> IO ()
  tanh         :: t -> t -> IO ()
  erf          :: t -> t -> IO ()
  erfinv       :: t -> t -> IO ()
  pow          :: t -> t -> HsReal t -> IO ()
  tpow         :: t -> HsReal t -> t -> IO ()
  sqrt         :: t -> t -> IO ()
  rsqrt        :: t -> t -> IO ()
  ceil         :: t -> t -> IO ()
  floor        :: t -> t -> IO ()
  round        :: t -> t -> IO ()
  trunc        :: t -> t -> IO ()
  frac         :: t -> t -> IO ()
  lerp         :: t -> t -> t -> HsReal t -> IO ()
  mean         :: t -> t -> Int32 -> Int32 -> IO ()
  std          :: t -> t -> Int32 -> Int32 -> Int32 -> IO ()
  var          :: t -> t -> Int32 -> Int32 -> Int32 -> IO ()
  norm         :: t -> t -> HsReal t -> Int32 -> Int32 -> IO ()
  renorm       :: t -> t -> HsReal t -> Int32 -> HsReal t -> IO ()
  dist         :: t -> t -> HsReal t -> IO (HsAccReal t)
  histc        :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  bhistc       :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  meanall      :: t -> IO (HsAccReal t)
  varall       :: t -> Int32 -> IO (HsAccReal t)
  stdall       :: t -> Int32 -> IO (HsAccReal t)
  normall      :: t -> HsReal t -> IO (HsAccReal t)
  linspace     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  logspace     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  rand         :: t -> Ptr CTHGenerator -> LongStorage -> IO ()
  randn        :: t -> Ptr CTHGenerator -> LongStorage -> IO ()



