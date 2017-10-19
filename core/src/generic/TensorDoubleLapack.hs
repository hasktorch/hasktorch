module TensorDoubleLapack (
  gesv
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

gesv a b = do
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

gesvd = undefined

gesvd2 = undefined

test = do
  rng <- newRNG
  let rnd = tdNew (D2 2 2)
  t <- uniformT rnd rng (-1.0) 1.0
  let b = tensorDoubleInit (D1 2) 1.0
  (resA, resB) <- gesv t b
  disp resA
  disp resB
  pure ()
