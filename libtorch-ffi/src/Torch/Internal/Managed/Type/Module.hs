
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}



module Torch.Internal.Managed.Type.Module where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Foreign.ForeignPtr.Unsafe
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import Control.Monad(forM)

import qualified Torch.Internal.Unmanaged.Type.Module as Unmanaged

newModule :: ForeignPtr StdString -> IO (ForeignPtr Module)
newModule = cast1 Unmanaged.newModule

save :: ForeignPtr Module -> FilePath -> IO ()
save = cast2 Unmanaged.save

load :: FilePath -> IO (ForeignPtr Module)
load = cast1 Unmanaged.load

forward :: ForeignPtr Module -> (ForeignPtr (StdVector IValue)) -> IO (ForeignPtr IValue)
forward = cast2 Unmanaged.forward

registerParameter :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr Tensor -> CBool -> IO ()
registerParameter = cast4 Unmanaged.registerParameter

registerModule :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr Module -> IO ()
registerModule = cast3 Unmanaged.registerModule

train :: ForeignPtr Module -> CBool -> IO ()
train = cast2 Unmanaged.train

runMethod :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr (C10List IValue) -> IO (Ptr IValue)
runMethod = cast3 Unmanaged.runMethod

runMethod1 :: ForeignPtr Module -> ForeignPtr StdString -> ForeignPtr IValue -> IO (Ptr IValue)
runMethod1 = cast3 Unmanaged.runMethod1

getParameters :: ForeignPtr Module -> IO (ForeignPtr TensorList)
getParameters = cast1 Unmanaged.getParameters

setParameters :: ForeignPtr Module -> ForeignPtr TensorList -> IO ()
setParameters = cast2 Unmanaged.setParameters

getNamedParameters :: ForeignPtr Module -> IO [(ForeignPtr StdString,ForeignPtr Tensor)]
getNamedParameters obj = withForeignPtr obj $ \obj' -> do
  v <- Unmanaged.getNamedParameters obj'
  forM v $ \(a,b) -> do
    a' <- uncast a return 
    b' <- uncast b return
    return (a',b')

getNamedBuffers :: ForeignPtr Module -> IO [(ForeignPtr StdString,ForeignPtr Tensor)]
getNamedBuffers obj = withForeignPtr obj $ \obj' -> do
  v <- Unmanaged.getNamedBuffers obj'
  forM v $ \(a,b) -> do
    a' <- uncast a return 
    b' <- uncast b return
    return (a',b')

getNamedAttributes :: ForeignPtr Module -> IO [(ForeignPtr StdString,ForeignPtr IValue)]
getNamedAttributes obj = withForeignPtr obj $ \obj' -> do
  v <- Unmanaged.getNamedAttributes obj'
  forM v $ \(a,b) -> do
    a' <- uncast a return 
    b' <- uncast b return
    return (a',b')

getNamedModules :: ForeignPtr Module -> IO [(ForeignPtr StdString,ForeignPtr Module)]
getNamedModules obj = withForeignPtr obj $ \obj' -> do
  v <- Unmanaged.getNamedModules obj'
  forM v $ \(a,b) -> do
    a' <- uncast a return 
    b' <- uncast b return
    return (a',b')

getNamedChildren :: ForeignPtr Module -> IO [(ForeignPtr StdString,ForeignPtr Module)]
getNamedChildren obj = withForeignPtr obj $ \obj' -> do
  v <- Unmanaged.getNamedChildren obj'
  forM v $ \(a,b) -> do
    a' <- uncast a return 
    b' <- uncast b return
    return (a',b')

toDevice :: ForeignPtr Module -> DeviceType -> Int16 -> IO ()
toDevice = cast3 Unmanaged.toDevice

clone :: ForeignPtr Module -> IO (ForeignPtr Module)
clone = cast1 Unmanaged.clone

define :: ForeignPtr Module -> ForeignPtr StdString -> IO ()
define = cast2 Unmanaged.define

-- TODO: Not using unsafeForeignPtrToPtr
trace :: String -> String -> (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> ForeignPtr TensorList -> IO (ForeignPtr Module)
trace moduleName functionName func inputs = cast3 (\m f inps -> Unmanaged.trace m f (trans func) inps) moduleName functionName inputs
  where
    trans :: (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> Ptr TensorList -> IO (Ptr TensorList)
    trans func inputs = do
      inputs' <- fromPtr inputs
      ret <- func inputs'
      return $ unsafeForeignPtrToPtr ret

traceAsGraph :: (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> ForeignPtr TensorList -> IO (ForeignPtr (SharedPtr JitGraph))
traceAsGraph func inputs = cast1 (\inps -> Unmanaged.traceAsGraph (trans func) inps) inputs
  where
    trans :: (ForeignPtr TensorList -> IO (ForeignPtr TensorList)) -> Ptr TensorList -> IO (Ptr TensorList)
    trans func inputs = do
      inputs' <- fromPtr inputs
      ret <- func inputs'
      return $ unsafeForeignPtrToPtr ret

printGraph :: ForeignPtr (SharedPtr JitGraph) -> IO (ForeignPtr StdString)
printGraph graph = cast1 Unmanaged.printGraph graph

printOnnx :: ForeignPtr (SharedPtr JitGraph) -> IO (ForeignPtr StdString)
printOnnx graph = cast1 Unmanaged.printOnnx graph

dumpToStr
  :: ForeignPtr Module
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr StdString)
dumpToStr = cast4 Unmanaged.dumpToStr
