 module Torch.Core.Tensor.Dynamic.DoubleLapack (
  td_gesv
  , td_gesv_
  , td_gels
  , td_gels_
  , td_getri
  , td_getri_
  , td_qr
  , td_qr_
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Dynamic.Double
import Torch.Core.Tensor.Dynamic.DoubleRandom
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import Torch.Core.Random

import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorLapack

td_gesv :: TensorDouble -> TensorDouble -> (TensorDouble, TensorDouble)
td_gesv a b = unsafePerformIO $ do
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

td_gesv_
  :: TensorDouble
     -> TensorDouble -> TensorDouble -> TensorDouble -> IO ()
td_gesv_ resA resB a b = do
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

td_gels :: TensorDouble -> TensorDouble -> (TensorDouble, TensorDouble)
td_gels a b = unsafePerformIO $ do
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

td_gels_
  :: TensorDouble
     -> TensorDouble -> TensorDouble -> TensorDouble -> IO ()
td_gels_ resA resB a b = do
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

td_getri :: TensorDouble -> TensorDouble
td_getri a = unsafePerformIO $ do
  let resA = td_new (tdDim a)
  withForeignPtr (tdTensor resA)
    (\resARaw ->
        withForeignPtr (tdTensor a)
        (\aRaw ->
            c_THDoubleTensor_getri resARaw aRaw
        )
    )
  pure resA

td_getri_ :: TensorDouble -> TensorDouble -> IO ()
td_getri_ resA a = do
  withForeignPtr (tdTensor resA)
    (\resARaw ->
        withForeignPtr (tdTensor a)
        (\aRaw ->
            c_THDoubleTensor_getri resARaw aRaw
        )
    )
  pure ()


td_qr :: TensorDouble -> (TensorDouble, TensorDouble)
td_qr a = unsafePerformIO $ do
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

td_qr_ :: TensorDouble -> TensorDouble -> TensorDouble -> IO ()
td_qr_ resQ resR a = do
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

td_gesvd = undefined

td_gesvd2 = undefined

