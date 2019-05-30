{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}


-----------------------------------------------------------------------------
-- |
-- Module      : ATen.Cast from FFICXX.Runtime.Cast
-- Copyright   : (c) 2011-2017 Ian-Woo Kim
--
-- License     : BSD3
-- Maintainer  : Junji Hashimoto<junji.hashimoto@gmail.com>
-- Stability   : experimental
-- Portability : GHC
--
-----------------------------------------------------------------------------

module ATen.Cast where

import Control.Monad         ((>=>))
import Data.ByteString.Char8 (ByteString,packCString, useAsCString)
import Data.String
import Data.Word
import Data.Int
import Foreign.C
import Foreign.C.String
import Foreign.ForeignPtr
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import ATen.Class


instance Castable () () where
  cast x f = f x
  uncast x f = f x

instance Castable CBool CBool where
  cast x f = f x
  uncast x f = f x

instance Castable CChar CChar where
  cast x f = f x
  uncast x f = f x

instance Castable CClock CClock where
  cast x f = f x
  uncast x f = f x

instance Castable CDouble CDouble where
  cast x f = f x
  uncast x f = f x

instance Castable CFile CFile where
  cast x f = f x
  uncast x f = f x

instance Castable CFloat CFloat where
  cast x f = f x
  uncast x f = f x

instance Castable CFpos CFpos where
  cast x f = f x
  uncast x f = f x

instance Castable CInt CInt where
  cast x f = f x
  uncast x f = f x

instance Castable CIntMax CIntMax where
  cast x f = f x
  uncast x f = f x

instance Castable CIntPtr CIntPtr where
  cast x f = f x
  uncast x f = f x

instance Castable CJmpBuf CJmpBuf where
  cast x f = f x
  uncast x f = f x

instance Castable CLLong CLLong where
  cast x f = f x
  uncast x f = f x

instance Castable CLong CLong where
  cast x f = f x
  uncast x f = f x

instance Castable CPtrdiff CPtrdiff where
  cast x f = f x
  uncast x f = f x

instance Castable CSChar CSChar where
  cast x f = f x
  uncast x f = f x

instance Castable CSUSeconds CSUSeconds where
  cast x f = f x
  uncast x f = f x

instance Castable CShort CShort where
  cast x f = f x
  uncast x f = f x

instance Castable CSigAtomic CSigAtomic where
  cast x f = f x
  uncast x f = f x

instance Castable CSize CSize where
  cast x f = f x
  uncast x f = f x

instance Castable CTime CTime where
  cast x f = f x
  uncast x f = f x

instance Castable CUChar CUChar where
  cast x f = f x
  uncast x f = f x

instance Castable CUInt CUInt where
  cast x f = f x
  uncast x f = f x

instance Castable CUIntMax CUIntMax where
  cast x f = f x
  uncast x f = f x

instance Castable CUIntPtr CUIntPtr where
  cast x f = f x
  uncast x f = f x

instance Castable CULLong CULLong where
  cast x f = f x
  uncast x f = f x

instance Castable CULong CULong where
  cast x f = f x
  uncast x f = f x

instance Castable CUSeconds CUSeconds where
  cast x f = f x
  uncast x f = f x

instance Castable CUShort CUShort where
  cast x f = f x
  uncast x f = f x

instance Castable CWchar CWchar where
  cast x f = f x
  uncast x f = f x

instance Castable Int8 Int8 where
  cast x f = f x
  uncast x f = f x

instance Castable Int16 Int16 where
  cast x f = f x
  uncast x f = f x

instance Castable Int32 Int32 where
  cast x f = f x
  uncast x f = f x

instance Castable Int64 Int64 where
  cast x f = f x
  uncast x f = f x

instance Castable Word8 Word8 where
  cast x f = f x
  uncast x f = f x

instance Castable Word16 Word16 where
  cast x f = f x
  uncast x f = f x

instance Castable Word32 Word32 where
  cast x f = f x
  uncast x f = f x

instance Castable Word64 Word64 where
  cast x f = f x
  uncast x f = f x

instance Castable Int CInt where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable Int16 CShort where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable Int8 CChar where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable Word CUInt where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable Word8 CChar where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable Double CDouble where
  cast x f = f (realToFrac x)
  uncast x f = f (realToFrac x)

instance Castable [Double] (Ptr CDouble) where
  cast xs f = newArray (map realToFrac xs) >>= f
  uncast xs f = undefined

instance Castable [Int] (Ptr CInt) where
  cast xs f = newArray (map fromIntegral xs) >>= f
  uncast xs f = undefined

instance Castable ByteString CString where
  cast x f = useAsCString x f
  uncast x f = packCString x >>= f

instance Castable [ByteString] (Ptr CString) where
  cast xs f = do ys <- mapM (\x -> useAsCString x return) xs
                 withArray ys $ \cptr -> f cptr
  uncast xs f = undefined

instance Castable (Ptr a) (Ptr a) where
  cast x f = f x
  uncast x f = f x

instance (CppObject a) => Castable (ForeignPtr a) (Ptr a) where
  cast x f = withForeignPtr x f
  uncast x f = fromPtr x >>= f

cast0 :: (Castable a ca) => (IO ca) -> IO a
cast0 f = f >>= \ca -> uncast ca return

cast1 :: (Castable a ca, Castable y cy)
       => (ca -> IO cy) -> a -> IO y
cast1 f a = cast a $ \ca -> f ca >>= \cy -> uncast cy return

cast2 :: (Castable a ca, Castable x1 cx1, Castable y cy)
       => (ca -> cx1 -> IO cy) -> a -> x1 -> IO y
cast2 f a x1 = cast a $ \ca ->
                  cast x1 $ \cx1 ->
                    f ca cx1 >>= \cy -> uncast cy return

cast3 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable y cy)
       => (ca -> cx1 -> cx2 -> IO cy) -> a -> x1 -> x2-> IO y
cast3 f a x1 x2 = cast a $ \ca ->
                     cast x1 $ \cx1 ->
                       cast x2 $ \cx2 ->
                         f ca cx1 cx2 >>= \cy -> uncast cy return

cast4 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> IO cy) -> a -> x1 -> x2 -> x3 -> IO y
cast4 f a x1 x2 x3 = cast a $ \ca ->
                        cast x1 $ \cx1 ->
                          cast x2 $ \cx2 ->
                            cast x3 $ \cx3 ->
                              f ca cx1 cx2 cx3 >>= \cy -> uncast cy return

cast5 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> IO cy) -> a -> x1 -> x2 -> x3 -> x4 -> IO y
cast5 f a x1 x2 x3 x4 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            f ca cx1 cx2 cx3 cx4 >>= \cy -> uncast cy return


cast6 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
           Castable x5 cx5, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> IO cy) -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> IO y
cast6 f a x1 x2 x3 x4 x5 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              f ca cx1 cx2 cx3 cx4 cx5 >>= \cy -> uncast cy return

cast7 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
           Castable x5 cx5, Castable x6 cx6, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> IO y
cast7 f a x1 x2 x3 x4 x5 x6 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                f ca cx1 cx2 cx3 cx4 cx5 cx6 >>= \cy -> uncast cy return

cast8 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
           Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> IO y
cast8 f a x1 x2 x3 x4 x5 x6 x7 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 >>= \cy -> uncast cy return


cast9 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
           Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> IO y
cast9 f a x1 x2 x3 x4 x5 x6 x7 x8 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 >>= \cy -> uncast cy return

cast10 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
           Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
           Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 -> IO y
cast10 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 >>= \cy -> uncast cy return

cast11 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
            Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
            Castable x10 cx10, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 -> cx10 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 -> x10 -> IO y
cast11 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      cast x10 $ \cx10 ->
                        f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 >>= \cy -> uncast cy return

cast12 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
            Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
            Castable x10 cx10, Castable x11 cx11, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 -> cx10 -> cx11 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 -> x10 -> x11 -> IO y
cast12 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      cast x10 $ \cx10 ->
                        cast x11 $ \cx11 ->
                         f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 >>= \cy -> uncast cy return


cast13 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
            Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
            Castable x10 cx10, Castable x11 cx11, Castable x12 cx12, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 -> cx10 -> cx11 -> cx12 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 -> x10 -> x11 -> x12 -> IO y
cast13 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      cast x10 $ \cx10 ->
                        cast x11 $ \cx11 ->
                          cast x12 $ \cx12 ->
                            f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12 >>= \cy -> uncast cy return

cast14 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
            Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
            Castable x10 cx10, Castable x11 cx11, Castable x12 cx12, Castable x13 cx13, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 -> cx10 -> cx11 -> cx12 -> cx13 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 -> x10 -> x11 -> x12 -> x13 -> IO y
cast14 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      cast x10 $ \cx10 ->
                        cast x11 $ \cx11 ->
                          cast x12 $ \cx12 ->
                          cast x13 $ \cx13 ->
                            f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12 cx13 >>= \cy -> uncast cy return


cast15 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
            Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
            Castable x10 cx10, Castable x11 cx11, Castable x12 cx12, Castable x13 cx13, Castable x14 cx14, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 -> cx10 -> cx11 -> cx12 -> cx13 -> cx14 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 -> x10 -> x11 -> x12 -> x13 -> x14 -> IO y
cast15 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      cast x10 $ \cx10 ->
                        cast x11 $ \cx11 ->
                          cast x12 $ \cx12 ->
                          cast x13 $ \cx13 ->
                          cast x14 $ \cx14 ->
                            f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12 cx13 cx14 >>= \cy -> uncast cy return


cast16 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
            Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
            Castable x10 cx10, Castable x11 cx11, Castable x12 cx12, Castable x13 cx13, Castable x14 cx14, Castable x15 cx15, Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 -> cx10 -> cx11 -> cx12 -> cx13 -> cx14 -> cx15 -> IO cy)
          -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 -> x10 -> x11 -> x12 -> x13 -> x14 -> x15 -> IO y
cast16 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      cast x10 $ \cx10 ->
                        cast x11 $ \cx11 ->
                          cast x12 $ \cx12 ->
                          cast x13 $ \cx13 ->
                          cast x14 $ \cx14 ->
                          cast x15 $ \cx15 ->
                            f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12 cx13 cx14 cx15 >>= \cy -> uncast cy return

cast21 :: (Castable a ca, Castable x1 cx1, Castable x2 cx2, Castable x3 cx3, Castable x4 cx4,
            Castable x5 cx5, Castable x6 cx6, Castable x7 cx7, Castable x8 cx8, Castable x9 cx9,
            Castable x10 cx10, Castable x11 cx11, Castable x12 cx12, Castable x13 cx13, Castable x14 cx14, Castable x15 cx15,
            Castable x16 cx16, Castable x17 cx17, Castable x18 cx18, Castable x19 cx19, Castable x20 cx20,
            Castable y cy)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> cx6 -> cx7 -> cx8 -> cx9 ->
           cx10 -> cx11 -> cx12 -> cx13 -> cx14 -> cx15 -> cx16 -> cx17 -> cx18 -> cx19 ->
           cx20 -> IO cy) ->
          a -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9 ->
          x10 -> x11 -> x12 -> x13 -> x14 -> x15 -> x16 -> x17 -> x18 -> x19 -> x20 -> IO y
cast21 f a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              cast x6 $ \cx6 ->
                cast x7 $ \cx7 ->
                  cast x8 $ \cx8 ->
                    cast x9 $ \cx9 ->
                      cast x10 $ \cx10 ->
                        cast x11 $ \cx11 ->
                          cast x12 $ \cx12 ->
                          cast x13 $ \cx13 ->
                          cast x14 $ \cx14 ->
                          cast x15 $ \cx15 ->
                          cast x16 $ \cx16 ->
                          cast x17 $ \cx17 ->
                          cast x18 $ \cx18 ->
                          cast x19 $ \cx19 ->
                          cast x20 $ \cx20 ->
                            f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9
                              cx10 cx11 cx12 cx13 cx14 cx15 cx16 cx17 cx18 cx19
                              cx20 >>= \cy -> uncast cy return

