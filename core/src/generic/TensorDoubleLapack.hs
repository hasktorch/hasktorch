 module TensorDoubleLapack (
  gesv
  , gesv_
  , qr
  , qr_
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import TensorDouble
import TensorDoubleRandom
import TensorRaw
import TensorTypes
import TensorUtils
import Random

import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorLapack

gesv :: TensorDouble -> TensorDouble -> (TensorDouble, TensorDouble)
gesv a b = unsafePerformIO $ do
  let resB = tdNew (tdDim a)
  let resA = tdNew (tdDim a)
  withForeignPtr (tdTensor resB)
    (\resBRaw ->
       withForeignPtr (tdTensor resA)
         (\resARaw ->
            withForeignPtr (tdTensor b)
              (\bRaw ->
                 withForeignPtr (tdTensor a)
                   (\aRaw ->
                      c_THDoubleTensor_gesv resBRaw resARaw bRaw aRaw
                   )
              )
         )
    )
  pure (resA, resB)

gesv_
  :: TensorDouble
     -> TensorDouble -> TensorDouble -> TensorDouble -> IO ()
gesv_ resA resB a b = do
  withForeignPtr (tdTensor resB)
    (\resBRaw ->
       withForeignPtr (tdTensor resA)
         (\resARaw ->
            withForeignPtr (tdTensor b)
              (\bRaw ->
                 withForeignPtr (tdTensor a)
                   (\aRaw ->
                      c_THDoubleTensor_gesv resBRaw resARaw bRaw aRaw
                   )
              )
         )
    )
  pure ()


qr :: TensorDouble -> (TensorDouble, TensorDouble)
qr a = unsafePerformIO $ do
  let resQ = tdNew (tdDim a)
  let resR = tdNew (tdDim a)
  withForeignPtr (tdTensor resQ)
    (\resQRaw ->
       withForeignPtr (tdTensor resR)
         (\resRRaw ->
             withForeignPtr (tdTensor a)
             (\aRaw ->
                 c_THDoubleTensor_qr resQRaw resRRaw aRaw
             )
         )
    )
  pure (resQ, resR)

qr_ :: TensorDouble -> TensorDouble -> TensorDouble -> IO ()
qr_ resQ resR a = do
  withForeignPtr (tdTensor resQ)
    (\resQRaw ->
       withForeignPtr (tdTensor resR)
         (\resRRaw ->
             withForeignPtr (tdTensor a)
             (\aRaw ->
                 c_THDoubleTensor_qr resQRaw resRRaw aRaw
             )
         )
    )
  pure ()

gesvd = undefined

gesvd2 = undefined

test = do
  rng <- newRNG
  let rnd = tdNew (D2 2 2)
  t <- uniformT rnd rng (-1.0) 1.0
  let b = tensorDoubleInit (D1 2) 1.0
  let (resA, resB) = gesv t b
  disp resA
  disp resB

  let (resQ, resR) = qr t
  disp resQ
  disp resR
  pure ()
