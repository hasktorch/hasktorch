{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}



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
import ATen.GC

instance Castable a a where
  cast x f = f x
  uncast x f = f x

instance Castable CBool Bool where
  cast x f = f (if x then 1 else 0)
  uncast x f = f (x /= 0)

instance Castable CInt Int where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable Int64 Int where
  cast x f = f (fromIntegral x)
  -- TODO: Int64 might have a wider range than Int
  uncast x f = f (fromIntegral x)

instance Castable CShort Int16 where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable CChar Int8 where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable CUInt Word where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable CChar Word8 where
  cast x f = f (fromIntegral x)
  uncast x f = f (fromIntegral x)

instance Castable CDouble Double where
  cast x f = f (realToFrac x)
  uncast x f = f (realToFrac x)

instance Castable (Ptr CDouble) [Double] where
  cast xs f = newArray (map realToFrac xs) >>= f
  uncast xs f = undefined

instance Castable (Ptr CInt) [Int] where
  cast xs f = newArray (map fromIntegral xs) >>= f
  uncast xs f = undefined

instance Castable CString ByteString where
  cast x f = useAsCString x f
  uncast x f = packCString x >>= f

instance Castable (Ptr CString) [ByteString] where
  cast xs f = do ys <- mapM (\x -> useAsCString x return) xs
                 withArray ys $ \cptr -> f cptr
  uncast xs f = undefined

instance (CppObject a) => Castable (Ptr a) (ForeignPtr a) where
  cast x f = withForeignPtr x f
  uncast x f = fromPtr x >>= f

--------------------------------------------------------------------------------
-- Tuples of Castable
--------------------------------------------------------------------------------

instance (Castable a' a, Castable b' b) => Castable (a',b') (a,b) where
  cast (t0,t1) f = do
    t0' <- cast t0 return
    t1' <- cast t1 return
    f (t0',t1')
  uncast (t0,t1) f = do
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    f (t0',t1')

instance (Castable a' a, Castable b' b, Castable c' c) => Castable (a',b',c') (a,b,c) where
  cast (t0,t1,t2) f = do
    t0' <- cast t0 return
    t1' <- cast t1 return
    t2' <- cast t2 return
    f (t0',t1',t2')
  uncast (t0,t1,t2) f = do
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    f (t0',t1',t2')

instance (Castable a' a, Castable b' b, Castable c' c, Castable d' d) => Castable (a',b',c',d') (a,b,c,d) where
  cast (t0,t1,t2,t3) f = do
    t0' <- cast t0 return
    t1' <- cast t1 return
    t2' <- cast t2 return
    t3' <- cast t3 return
    f (t0',t1',t2',t3')
  uncast (t0,t1,t2,t3) f = do
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    t3' <- uncast t3 return
    f (t0',t1',t2',t3')

instance (Castable a' a, Castable b' b, Castable c' c, Castable d' d, Castable e' e) => Castable (a',b',c',d',e') (a,b,c,d,e) where
  cast (t0,t1,t2,t3,t4) f = do
    t0' <- cast t0 return
    t1' <- cast t1 return
    t2' <- cast t2 return
    t3' <- cast t3 return
    t4' <- cast t4 return
    f (t0',t1',t2',t3',t4')
  uncast (t0,t1,t2,t3,t4) f = do
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    t3' <- uncast t3 return
    t4' <- uncast t4 return
    f (t0',t1',t2',t3',t4')

instance (Castable a' a, Castable b' b, Castable c' c,
          Castable d' d, Castable e' e, Castable f' f) => Castable (a',b',c',d',e',f') (a,b,c,d,e,f) where
  cast (t0,t1,t2,t3,t4,t5) f = do
    t0' <- cast t0 return
    t1' <- cast t1 return
    t2' <- cast t2 return
    t3' <- cast t3 return
    t4' <- cast t4 return
    t5' <- cast t5 return
    f (t0',t1',t2',t3',t4',t5')
  uncast (t0,t1,t2,t3,t4,t5) f = do
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    t3' <- uncast t3 return
    t4' <- uncast t4 return
    t5' <- uncast t5 return
    f (t0',t1',t2',t3',t4',t5')

--------------------------------------------------------------------------------
-- These casts convert the value from C++ Tuple(CppTuple) to Haskell Tuple.
-- Reverse side is not supported.
--------------------------------------------------------------------------------

instance (CppTuple2 c, Castable (A c) a, Castable (B c) b) => Castable c (a,b) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    f (t0',t1')

instance (CppTuple3 d, Castable (A d) a, Castable (B d) b, Castable (C d) c) => Castable d (a,b,c) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    f (t0',t1',t2')

instance (CppTuple4 e, Castable (A e) a, Castable (B e) b, Castable (C e) c, Castable (D e) d) => Castable e (a,b,c,d) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    t3' <- uncast t3 return
    f (t0',t1',t2',t3')

instance (CppTuple5 f, Castable (A f) a, Castable (B f) b, Castable (C f) c, Castable (D f) d, Castable (E f) e) => Castable f (a,b,c,d,e) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    t4 <- get4 t
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    t3' <- uncast t3 return
    t4' <- uncast t4 return
    f (t0',t1',t2',t3',t4')

instance (CppTuple6 g,
          Castable (A g) a, Castable (B g) b, Castable (C g) c,
          Castable (D g) d, Castable (E g) e, Castable (F g) f) => Castable g (a,b,c,d,e,f) where
  cast _ _ = undefined
  uncast t f = do
    t0 <- get0 t
    t1 <- get1 t
    t2 <- get2 t
    t3 <- get3 t
    t4 <- get4 t
    t5 <- get5 t
    t0' <- uncast t0 return
    t1' <- uncast t1 return
    t2' <- uncast t2 return
    t3' <- uncast t3 return
    t4' <- uncast t4 return
    t5' <- uncast t5 return
    f (t0',t1',t2',t3',t4',t5')

--------------------------------------------------------------------------------
-- Cast functions for various numbers of arguments
--------------------------------------------------------------------------------

cast0 :: (Castable ca a) => (IO ca) -> IO a
cast0 f = retryWithGC (f) >>= \ca -> uncast ca return

cast1 :: (Castable ca a, Castable cy y)
       => (ca -> IO cy) -> a -> IO y
cast1 f a = cast a $ \ca -> retryWithGC (f ca) >>= \cy -> uncast cy return

cast2 :: (Castable ca a, Castable cx1 x1, Castable cy y)
       => (ca -> cx1 -> IO cy) -> a -> x1 -> IO y
cast2 f a x1 = cast a $ \ca ->
                  cast x1 $ \cx1 ->
                    retryWithGC (f ca cx1) >>= \cy -> uncast cy return

cast3 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cy y)
       => (ca -> cx1 -> cx2 -> IO cy) -> a -> x1 -> x2-> IO y
cast3 f a x1 x2 = cast a $ \ca ->
                     cast x1 $ \cx1 ->
                       cast x2 $ \cx2 ->
                         retryWithGC (f ca cx1 cx2) >>= \cy -> uncast cy return

cast4 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cy y)
       => (ca -> cx1 -> cx2 -> cx3 -> IO cy) -> a -> x1 -> x2 -> x3 -> IO y
cast4 f a x1 x2 x3 = cast a $ \ca ->
                        cast x1 $ \cx1 ->
                          cast x2 $ \cx2 ->
                            cast x3 $ \cx3 ->
                              retryWithGC (f ca cx1 cx2 cx3) >>= \cy -> uncast cy return

cast5 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4, Castable cy y)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> IO cy) -> a -> x1 -> x2 -> x3 -> x4 -> IO y
cast5 f a x1 x2 x3 x4 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            retryWithGC (f ca cx1 cx2 cx3 cx4) >>= \cy -> uncast cy return


cast6 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
           Castable cx5 x5, Castable cy y)
       => (ca -> cx1 -> cx2 -> cx3 -> cx4 -> cx5 -> IO cy) -> a -> x1 -> x2 -> x3 -> x4 -> x5 -> IO y
cast6 f a x1 x2 x3 x4 x5 =
  cast a $ \ca ->
    cast x1 $ \cx1 ->
      cast x2 $ \cx2 ->
        cast x3 $ \cx3 ->
          cast x4 $ \cx4 ->
            cast x5 $ \cx5 ->
              retryWithGC (f ca cx1 cx2 cx3 cx4 cx5) >>= \cy -> uncast cy return

cast7 :: (Castable ca a , Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
           Castable cx5 x5, Castable cx6 x6, Castable cy y)
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
                retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6) >>= \cy -> uncast cy return

cast8 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
           Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cy y)
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
                  retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7) >>= \cy -> uncast cy return


cast9 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
           Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cy y)
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
                    retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8) >>= \cy -> uncast cy return

cast10 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
           Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
           Castable cy y)
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
                      retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9) >>= \cy -> uncast cy return

cast11 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
            Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
            Castable cx10 x10, Castable cy y)
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
                        retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10) >>= \cy -> uncast cy return

cast12 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
            Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
            Castable cx10 x10, Castable cx11 x11, Castable cy y)
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
                         retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11) >>= \cy -> uncast cy return


cast13 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
            Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
            Castable cx10 x10, Castable cx11 x11, Castable cx12 x12, Castable cy y)
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
                            retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12) >>= \cy -> uncast cy return

cast14 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
            Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
            Castable cx10 x10, Castable cx11 x11, Castable cx12 x12, Castable cx13 x13, Castable cy y)
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
                            retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12 cx13) >>= \cy -> uncast cy return


cast15 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
            Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
            Castable cx10 x10, Castable cx11 x11, Castable cx12 x12, Castable cx13 x13, Castable cx14 x14, Castable cy y)
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
                            retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12 cx13 cx14) >>= \cy -> uncast cy return


cast16 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
            Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
            Castable cx10 x10, Castable cx11 x11, Castable cx12 x12, Castable cx13 x13, Castable cx14 x14, Castable cx15 x15, Castable cy y)
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
                            retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9 cx10 cx11 cx12 cx13 cx14 cx15) >>= \cy -> uncast cy return

cast21 :: (Castable ca a, Castable cx1 x1, Castable cx2 x2, Castable cx3 x3, Castable cx4 x4,
            Castable cx5 x5, Castable cx6 x6, Castable cx7 x7, Castable cx8 x8, Castable cx9 x9,
            Castable cx10 x10, Castable cx11 x11, Castable cx12 x12, Castable cx13 x13, Castable cx14 x14, Castable cx15 x15,
            Castable cx16 x16, Castable cx17 x17, Castable cx18 x18, Castable cx19 x19, Castable cx20 x20,
            Castable cy y)
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
                            retryWithGC (f ca cx1 cx2 cx3 cx4 cx5 cx6 cx7 cx8 cx9
                              cx10 cx11 cx12 cx13 cx14 cx15 cx16 cx17 cx18 cx19
                              cx20) >>= \cy -> uncast cy return

