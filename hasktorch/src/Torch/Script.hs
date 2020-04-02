{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Torch.Script where

import Control.Monad (forM_, forM, replicateM)
import Control.Exception.Safe (throwIO)
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types
import System.IO.Unsafe
import Data.Int (Int16, Int64)
import Data.Word (Word8)
import Data.List (intercalate)
import Data.Proxy
import Data.Reflection
import Numeric

import Torch.Internal.Cast
import Torch.Internal.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..), CppObject(..))
import qualified Torch.Internal.Unmanaged.Type.Tensor as Unmanaged (tensor_data_ptr)
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Managed.Type.StdArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Cast as ATen
import qualified Torch.Internal.Type as ATen
import qualified Torch.Internal.Const as ATen
import Torch.Internal.Unmanaged.Type.IValue (IValueLike(..))
import Torch.Internal.Unmanaged.Type.C10Dict
import Torch.Internal.Managed.Type.IValue
import Torch.Internal.Type (TensorList)
import qualified Torch.Internal.Managed.Type.Module as LibTorch

import Torch.Device
import Torch.DType
import Torch.Tensor (Tensor(..))
import Torch.TensorOptions
import Torch.NN
import Torch.Autograd

newtype ScriptModule = UnsafeScriptModule (ForeignPtr ATen.Module)
newtype RawModule = UnsafeRawModule (ForeignPtr ATen.Module)

type RawIValue = ForeignPtr ATen.IValue
newtype Blob = UnsafeBlob (ForeignPtr (ATen.C10Ptr ATen.Blob))
newtype Object = UnsafeObject (ForeignPtr (ATen.C10Ptr ATen.IVObject))
newtype Future = UnsafeFuture (ForeignPtr (ATen.C10Ptr ATen.IVFuture))
newtype Capsule = UnsafeCapsule (ForeignPtr (ATen.C10Ptr ATen.Capsule))

instance Show Blob where
  show _ = "Blob"

instance Show Future where
  show _ = "Future"

instance Show Object where
  show _ = "Object"

instance Show Capsule where
  show _ = "Capsule"

data IValue
  = IVNone
  | IVTensor Tensor
  | IVDouble Double
  | IVInt Int64
  | IVBool Bool
  | IVTuple [IValue]
  | IVIntList [Int64]
  | IVDoubleList [Double]
  | IVBoolList [Bool]
  | IVString String
  | IVTensorList [Tensor]
  | IVBlob -- Blob
  | IVGenericList [IValue]
  | IVGenericDict [(IValue,IValue)]
  | IVFuture -- Future
  | IVDevice -- Device
  | IVObject -- Object
  | IVUninitialized
  | IVCapsule -- Capsule
  deriving (Show)

instance Castable ScriptModule (ForeignPtr ATen.Module) where
  cast (UnsafeScriptModule obj) f = f obj
  uncast obj f = f $ UnsafeScriptModule obj

instance Castable RawModule (ForeignPtr ATen.Module) where
  cast (UnsafeRawModule obj) f = f obj
  uncast obj f = f $ UnsafeRawModule obj

newModule :: String -> IO RawModule
newModule = cast1 LibTorch.newModule

save :: ScriptModule -> FilePath -> IO ()
save = cast2 LibTorch.save

save' :: RawModule -> FilePath -> IO ()
save' = cast2 LibTorch.save

data LoadMode
  = WithoutRequiredGrad
  | WithRequiredGrad
  deriving (Show,Eq)

load :: LoadMode -> FilePath -> IO ScriptModule
load WithoutRequiredGrad file = cast1 LibTorch.load file
load WithRequiredGrad file = do
  module'@(UnsafeRawModule rmodule) <- cast1 LibTorch.load file
  params <- getParametersIO module'
  paramsWithRequiredGrad <- forM params makeIndependent
  setParameters module' (map toDependent paramsWithRequiredGrad)
  return (UnsafeScriptModule rmodule)
  
load' :: FilePath -> IO RawModule
load' = cast1 LibTorch.load

forwardIO :: ScriptModule -> [IValue] -> IO IValue
forwardIO module' inputs' = cast2 forward' module' inputs'
  where
    forward' :: ScriptModule -> [RawIValue] -> IO RawIValue
    forward' = cast2 LibTorch.forward

forward :: ScriptModule -> [IValue] -> IValue
forward module' inputs' = unsafePerformIO $ forwardIO module' inputs'

registerParameter :: RawModule -> String -> Tensor -> Bool -> IO ()
registerParameter = cast4 LibTorch.registerParameter

registerModule :: RawModule -> String -> RawModule -> IO ()
registerModule = cast3 LibTorch.registerModule

getParameters :: ScriptModule -> [Tensor]
getParameters module' = unsafePerformIO $ cast1 LibTorch.getParameters module'

getParametersIO :: RawModule -> IO [Tensor]
getParametersIO module' = cast1 LibTorch.getParameters module'

setParameters :: RawModule -> [Tensor] -> IO ()
setParameters = cast2 LibTorch.setParameters

updateParameters :: LoadMode -> ScriptModule -> [Tensor] -> ScriptModule
updateParameters mode module' inputs = unsafePerformIO $ do
  case mode of
    WithoutRequiredGrad -> cast1 LibTorch.clone module'
    WithRequiredGrad -> do
      r <- cast1 LibTorch.clone module'
      paramsWithRequiredGrad <- forM inputs makeIndependent
      setParameters' r (map toDependent paramsWithRequiredGrad)
      return r
  where
    setParameters' :: ScriptModule -> [Tensor] -> IO ()
    setParameters' = cast2 LibTorch.setParameters

toScriptModule :: RawModule -> IO ScriptModule
toScriptModule rawModule = do
  (UnsafeRawModule r) <- clone rawModule
  return $ UnsafeScriptModule r

toRawModule :: ScriptModule -> IO RawModule
toRawModule scriptModule = do
  (UnsafeScriptModule r) <- clone' scriptModule
  return $ UnsafeRawModule r
  where
    clone' = cast1 LibTorch.clone

clone :: RawModule -> IO RawModule
clone = cast1 LibTorch.clone

train :: RawModule -> Bool -> IO ()
train = cast2 LibTorch.train

define :: RawModule -> String -> IO ()
define = cast2 LibTorch.define

runMethod :: ScriptModule -> String -> [IValue] -> IValue
runMethod module' func inputs = unsafePerformIO $ cast3 runMethod' module' func inputs 
  where
    runMethod' :: ScriptModule -> String -> [RawIValue] -> IO RawIValue
    runMethod' = cast3 LibTorch.runMethod

runMethod1 :: ScriptModule -> String -> IValue -> IValue
runMethod1 module' func input = unsafePerformIO $ cast3 runMethod1' module' func input
  where
    runMethod1' :: ScriptModule -> String -> RawIValue -> IO RawIValue
    runMethod1' = cast3 LibTorch.runMethod1

instance Parameterized ScriptModule where
  flattenParameters module' = map IndependentTensor $ getParameters module'
  replaceOwnParameters module' = do
    let len = length (getParameters module')
    ps' <- replicateM len nextParameter
    return $ updateParameters WithRequiredGrad module' (map toDependent ps')

trace :: String -> String -> ([Tensor] -> IO [Tensor]) -> [Tensor] -> IO RawModule
trace moduleName functionName func inputs = cast3 (\m f inps -> LibTorch.trace m f (trans func) inps) moduleName functionName inputs
  where
    trans :: ([Tensor] -> IO [Tensor]) -> ForeignPtr TensorList -> IO (ForeignPtr TensorList)
    trans func inputs =
      uncast inputs $ \inputs' -> do
        ret <- func inputs'
        cast ret return

traceWithParameters :: Parameterized f => String -> (f -> [Tensor] -> IO [Tensor]) -> f -> [Tensor] -> IO RawModule
traceWithParameters moduleName func parameterized_parameters inputs = do
  let parameters = (map toDependent) (flattenParameters parameterized_parameters)
      fromParams params = replaceParameters parameterized_parameters (map IndependentTensor params)
      plen = length parameters
      ilen = length inputs
  r <- trace moduleName "forwardWithParameters"
         (\parametersAndInputs ->
            func
              (fromParams (take plen parametersAndInputs))
              (drop plen parametersAndInputs)
         )
         (parameters++inputs)
  forM_ (zip [0..] parameters) $ \(i,p) ->
    registerParameter r ("p" ++ show i) p False
  let args = intercalate ", " $ map (\i ->  "i" ++ show i) [0..(ilen-1)]
      params = intercalate ", " $ map (\i ->  "self.p" ++ show i) [0..(plen-1)]
  define r $
    "def forward(self, " ++ args ++ "):\n" ++ 
    "    return self.forwardWithParameters(" ++ params ++ ", " ++ args ++ " )\n"
  return r

instance Castable [IValue] [RawIValue] where
  cast a f = (forM a $ \v -> cast v return) >>= f
  uncast a f = (forM a $ \v -> uncast v return) >>= f

instance Castable IValue RawIValue where
  cast (IVNone) f = newIValue >>= f
  cast (IVTensor (Unsafe v)) f = toIValue v>>= f
  cast (IVDouble v) f = toIValue v >>= f
  cast (IVInt v) f = toIValue v >>= f
  cast (IVBool v) f = toIValue v >>= f
  cast (IVTuple v) f = do
    rawIValues <- cast v return :: IO [RawIValue]
    c10tuple <- cast rawIValues return :: IO (ForeignPtr (ATen.C10Ptr ATen.IVTuple))
    f =<< toIValue c10tuple
  cast (IVIntList v) f = do
    v' <- cast v return :: IO (ForeignPtr (ATen.C10List Int64))
    f =<< toIValue v'
  cast (IVDoubleList v) f = do
    cdoubles <- forM v (flip cast return) :: IO [CDouble]
    c10list <- cast cdoubles return :: IO (ForeignPtr (ATen.C10List CDouble))
    f =<< toIValue c10list
  cast (IVBoolList v) f = do
    cbools <- forM v (flip cast return) :: IO [CBool]
    c10list <- cast cbools return :: IO (ForeignPtr (ATen.C10List CBool))
    f =<< toIValue c10list
  cast (IVString v) f = do
    v' <- cast v return :: IO (ForeignPtr (ATen.StdString))
    f =<< toIValue v'
  cast (IVTensorList v) f = do
    v' <- cast v return :: IO (ForeignPtr (ATen.C10List ATen.Tensor))
    f =<< toIValue v'
  cast (IVGenericList v) f = do
    rawIValues <- cast v return :: IO [RawIValue]
    c10list <- cast rawIValues return :: IO (ForeignPtr (ATen.C10List ATen.IValue))
    f =<< toIValue c10list
  cast (IVGenericDict v) f = do
    keys <- cast (map fst v) return :: IO [RawIValue]
    values <- cast (map snd v) return :: IO [RawIValue]
    let rawIValues = zip keys values
    c10list <- cast rawIValues return :: IO (ForeignPtr (ATen.C10Dict '(ATen.IValue,ATen.IValue)))
    f =<< toIValue c10list
--  cast (IVBlob (UnsafeBlob v)) f = toIValue v >>= f
--  cast (IVFuture (UnsafeFuture v)) f = toIValue v >>= f
--  cast (IVDevice v) f = toIValue v >>= f
--  cast (IVObject (UnsafeObject v)) f = toIValue v >>= f
--  cast (IVUninitialized) f = f (toIValue v)
--  cast (IVCapsule v) f = toIValue v >>= f
  cast a f = throwIO $ userError $ "Unsupported data-type:" ++ show a
  uncast obj f =
    select
      [ (iValue_isNone obj, f IVNone)
      , (iValue_isTensor obj, fromIValue obj >>= f . IVTensor . Unsafe)
      , (iValue_isDouble obj, fromIValue obj >>= f . IVDouble)
      , (iValue_isInt obj, fromIValue obj >>= f . IVInt)
      , (iValue_isBool obj, fromIValue obj >>= f . IVBool)
      , (iValue_isString obj, do
           v <- fromIValue obj :: IO (ForeignPtr ATen.StdString)
           str <- uncast v return :: IO String
           f (IVString str)
        )
      , (iValue_isTensorList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List ATen.Tensor))
           ts <- uncast v' return :: IO [Tensor]
           f (IVTensorList ts)
        )
      , (iValue_isDoubleList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List CDouble))
           cdoubles <- uncast v' return :: IO [CDouble]
           doubles <- forM cdoubles (flip uncast return) :: IO [Double]
           f (IVDoubleList doubles)
        )
      , (iValue_isIntList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List Int64))
           ts <- uncast v' return :: IO [Int64]
           f (IVIntList ts)
        )
      , (iValue_isBoolList obj, do
           v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List CBool))
           cbools <- uncast v' return :: IO [CBool]
           bools <- forM cbools (flip uncast return) :: IO [Bool]
           f (IVBoolList bools)
        )
      , (iValue_isTuple obj, do
           c10tuple <- fromIValue obj :: IO (ForeignPtr (ATen.C10Ptr ATen.IVTuple))
           rawIValues <- uncast c10tuple return :: IO [RawIValue]
           ts <- uncast rawIValues return :: IO [IValue]
           f (IVTuple ts)
        )
      , (iValue_isGenericList obj, do
           c10list <- fromIValue obj :: IO (ForeignPtr (ATen.C10List ATen.IValue))
           rawIValues <- uncast c10list return :: IO [RawIValue]
           ts <- uncast rawIValues return :: IO [IValue]
           f (IVGenericList ts)
        )
      , (iValue_isGenericDict obj, do
           c10list <- fromIValue obj :: IO (ForeignPtr (ATen.C10Dict '(ATen.IValue,ATen.IValue)))
           rawIValues <- uncast c10list return :: IO [(RawIValue,RawIValue)]
           ts <- forM rawIValues $ \(a,b) -> do
             a' <- uncast a return
             b' <- uncast b return
             return (a',b')
           f (IVGenericDict ts)
        )
      , (iValue_isBlob obj, f IVBlob)
      , (iValue_isFuture obj, f IVFuture)
      , (iValue_isDevice obj, f IVDevice)
      , (iValue_isObject obj, f IVObject)
      , (iValue_isCapsule obj, f IVCapsule)
      ]
    where
      select [] = throwIO $ userError "Unsupported IValue"
      select ((cond,body):xs) =
        cond >>= \case
          1 -> body
          _ -> select xs
