 module TensorDoubleLapack (
  gesv
  , gesv_
  , gels
  , gels_
  , getri
  , getri_
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
  let resB = td_new (tdDim a)
  let resA = td_new (tdDim a)
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

gels :: TensorDouble -> TensorDouble -> (TensorDouble, TensorDouble)
gels a b = unsafePerformIO $ do
  let resB = td_new (tdDim a)
  let resA = td_new (tdDim a)
  withForeignPtr (tdTensor resB)
    (\resBRaw ->
       withForeignPtr (tdTensor resA)
         (\resARaw ->
            withForeignPtr (tdTensor b)
              (\bRaw ->
                 withForeignPtr (tdTensor a)
                   (\aRaw ->
                      c_THDoubleTensor_gels resBRaw resARaw bRaw aRaw
                   )
              )
         )
    )
  pure (resA, resB)

gels_
  :: TensorDouble
     -> TensorDouble -> TensorDouble -> TensorDouble -> IO ()
gels_ resA resB a b = do
  withForeignPtr (tdTensor resB)
    (\resBRaw ->
       withForeignPtr (tdTensor resA)
         (\resARaw ->
            withForeignPtr (tdTensor b)
              (\bRaw ->
                 withForeignPtr (tdTensor a)
                   (\aRaw ->
                      c_THDoubleTensor_gels resBRaw resARaw bRaw aRaw
                   )
              )
         )
    )
  pure ()

getri :: TensorDouble -> TensorDouble
getri a = unsafePerformIO $ do
  let resA = td_new (tdDim a)
  withForeignPtr (tdTensor resA)
    (\resARaw ->
        withForeignPtr (tdTensor a)
        (\aRaw ->
            c_THDoubleTensor_getri resARaw aRaw
        )
    )
  pure resA

getri_ :: TensorDouble -> TensorDouble -> IO ()
getri_ resA a = do
  withForeignPtr (tdTensor resA)
    (\resARaw ->
        withForeignPtr (tdTensor a)
        (\aRaw ->
            c_THDoubleTensor_getri resARaw aRaw
        )
    )
  pure ()


qr :: TensorDouble -> (TensorDouble, TensorDouble)
qr a = unsafePerformIO $ do
  let resQ = td_new (tdDim a)
  let resR = td_new (tdDim a)
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
  let rnd = td_new (D2 2 2)
  t <- td_uniform rnd rng (-1.0) 1.0
  let b = td_init (D1 2) 1.0
  let (resA, resB) = gesv t b
  disp resA
  disp resB

  let (resQ, resR) = qr t
  disp resQ
  disp resR
  pure ()
