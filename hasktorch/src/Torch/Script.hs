{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Script where

import Control.Exception.Safe (throwIO)
import Control.Monad (forM, forM_, replicateM)
import Data.Int (Int16, Int64)
import Data.List (intercalate)
import Data.Proxy
import Data.Reflection
import Data.Word (Word8)
import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Numeric
import System.IO.Unsafe
import Torch.Autograd
import Torch.DType
import Torch.Device
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..), CppObject (..), CppTuple2 (..), CppTuple3 (..), CppTuple4 (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Cast as ATen
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import Torch.Internal.Managed.Type.IValue
import qualified Torch.Internal.Managed.Type.Module as LibTorch
import qualified Torch.Internal.Managed.Type.StdArray as ATen
import qualified Torch.Internal.Managed.Type.StdString as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import Torch.Internal.Type (TensorList)
import qualified Torch.Internal.Type as ATen
import Torch.Internal.Unmanaged.Type.C10Dict
import Torch.Internal.Unmanaged.Type.IValue (IValueLike (..))
import qualified Torch.Internal.Unmanaged.Type.Module as Unmanaged
import Torch.NN
import Torch.Tensor (Tensor (..), toDevice)
import Torch.TensorOptions

newtype ScriptModule = UnsafeScriptModule (ForeignPtr ATen.Module)

newtype RawModule = UnsafeRawModule (ForeignPtr ATen.Module)

instance Show ScriptModule where
  show obj = unsafePerformIO $ dumpToStr' obj

type RawIValue = ForeignPtr ATen.IValue

newtype Blob = UnsafeBlob (ForeignPtr (ATen.C10Ptr ATen.Blob))

newtype Object = UnsafeObject (ForeignPtr (ATen.C10Ptr ATen.IVObject))

newtype Future = UnsafeFuture (ForeignPtr (ATen.C10Ptr ATen.IVFuture))

newtype Capsule = UnsafeCapsule (ForeignPtr (ATen.C10Ptr ATen.Capsule))

-- | See https://github.com/pytorch/pytorch/wiki/PyTorch-IR
newtype Graph = UnsafeGraph (ForeignPtr (ATen.SharedPtr ATen.JitGraph))

data JitGraph = JitGraph
  { graphInputs :: [JitValue],
    graphOutputs :: [JitValue],
    graphNodes :: [JitNode]
  }
  deriving (Show, Eq)

data JitNode = JitNode
  { nodeInputs :: [JitValue],
    nodeOutputs :: [JitValue],
    nodeKind :: String
  }
  deriving (Show, Eq)

data JitValue = JitValue
  { valueId :: Int,
    valueType :: String
  }
  deriving (Show, Eq)

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
  | IVGenericDict [(IValue, IValue)]
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

instance Castable Graph (ForeignPtr (ATen.SharedPtr ATen.JitGraph)) where
  cast (UnsafeGraph obj) f = f obj
  uncast obj f = f $ UnsafeGraph obj

newModule :: String -> IO RawModule
newModule = cast1 LibTorch.newModule

saveScript :: ScriptModule -> FilePath -> IO ()
saveScript = cast2 LibTorch.save

saveScript' :: RawModule -> FilePath -> IO ()
saveScript' = cast2 LibTorch.save

data LoadMode
  = WithoutRequiredGrad
  | WithRequiredGrad
  deriving (Show, Eq)

-- | Load a torchscript file
loadScript :: LoadMode -> FilePath -> IO ScriptModule
loadScript WithoutRequiredGrad file = cast1 LibTorch.load file
loadScript WithRequiredGrad file = do
  module'@(UnsafeRawModule rmodule) <- cast1 LibTorch.load file
  params <- getParametersIO module'
  paramsWithRequiredGrad <- forM params makeIndependent
  setParameters module' (map toDependent paramsWithRequiredGrad)
  return (UnsafeScriptModule rmodule)

loadScript' :: FilePath -> IO RawModule
loadScript' = cast1 LibTorch.load

instance HasForward ScriptModule [IValue] IValue where
  forward module' = unsafePerformIO . forwardStoch module'
  forwardStoch = cast2 forward'
    where
      forward' :: ScriptModule -> [RawIValue] -> IO RawIValue
      forward' = cast2 LibTorch.forward

registerParameter :: RawModule -> String -> Tensor -> Bool -> IO ()
registerParameter = cast4 LibTorch.registerParameter

registerModule :: RawModule -> String -> RawModule -> IO ()
registerModule = cast3 LibTorch.registerModule

getParameters ::
  -- | module
  ScriptModule ->
  -- | output
  [Tensor]
getParameters = unsafePerformIO . cast1 LibTorch.getParameters

getParametersIO ::
  -- | module
  RawModule ->
  -- | output
  IO [Tensor]
getParametersIO = cast1 LibTorch.getParameters

setParameters :: RawModule -> [Tensor] -> IO ()
setParameters = cast2 LibTorch.setParameters

updateParameters :: LoadMode -> ScriptModule -> [Tensor] -> ScriptModule
updateParameters mode module' inputs = unsafePerformIO $
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

getNamedParameters ::
  -- | module
  ScriptModule ->
  -- | output
  [(String, Tensor)]
getNamedParameters (UnsafeScriptModule m) = unsafePerformIO $ do
  dat <- LibTorch.getNamedParameters m
  forM dat $ \(key, value) ->
    (,) <$> uncast key return <*> uncast value return

getNamedBuffers ::
  -- | module
  ScriptModule ->
  -- | output
  [(String, Tensor)]
getNamedBuffers (UnsafeScriptModule m) = unsafePerformIO $ do
  dat <- LibTorch.getNamedBuffers m
  forM dat $ \(key, value) ->
    (,) <$> uncast key return <*> uncast value return

-- | Load all attributes including training flags
-- This function returns IVObject type as Tensor type.
-- To get Tensor type, use get getNamedParameters and getNamedBuffers.
getNamedAttributes ::
  -- | module
  ScriptModule ->
  -- | output
  [(String, IValue)]
getNamedAttributes (UnsafeScriptModule m) = unsafePerformIO $ do
  dat <- LibTorch.getNamedAttributes m
  forM dat $ \(key, value) ->
    (,) <$> uncast key return <*> uncast value return

getNamedModules ::
  -- | module
  ScriptModule ->
  -- | output
  [(String, ScriptModule)]
getNamedModules (UnsafeScriptModule m) = unsafePerformIO $ do
  dat <- LibTorch.getNamedModules m
  forM dat $ \(key, value) ->
    (,) <$> uncast key return <*> uncast value return

getNamedChildren ::
  -- | module
  ScriptModule ->
  -- | output
  [(String, ScriptModule)]
getNamedChildren (UnsafeScriptModule m) = unsafePerformIO $ do
  dat <- LibTorch.getNamedChildren m
  forM dat $ \(key, value) ->
    (,) <$> uncast key return <*> uncast value return

toScriptModule :: RawModule -> IO ScriptModule
toScriptModule rawModule = do
  (UnsafeRawModule r) <- cloneRawModule rawModule
  return $ UnsafeScriptModule r

toRawModule :: ScriptModule -> IO RawModule
toRawModule scriptModule = do
  (UnsafeScriptModule r) <- clone' scriptModule
  return $ UnsafeRawModule r
  where
    clone' = cast1 LibTorch.clone

cloneRawModule :: RawModule -> IO RawModule
cloneRawModule = cast1 LibTorch.clone

data RuntimeMode = Eval | Train deriving (Show, Eq)

setRuntimeMode :: RawModule -> RuntimeMode -> IO ()
setRuntimeMode rmod mode = cast2 LibTorch.train rmod (mode == Train)

define :: RawModule -> String -> IO ()
define = cast2 LibTorch.define

dumpToStr ::
  -- | module
  ScriptModule ->
  -- | print_method_bodies
  Bool ->
  -- | print_attr_values
  Bool ->
  -- | print_param_values
  Bool ->
  -- | ouput
  IO String
dumpToStr = cast4 LibTorch.dumpToStr

dumpToStr' :: ScriptModule -> IO String
dumpToStr' obj = dumpToStr obj True True True

runMethod ::
  -- | module
  ScriptModule ->
  -- | func
  String ->
  -- | inputs
  [IValue] ->
  -- | output
  IValue
runMethod module' func = unsafePerformIO . cast3 runMethod' module' func
  where
    runMethod' :: ScriptModule -> String -> [RawIValue] -> IO RawIValue
    runMethod' = cast3 LibTorch.runMethod

runMethod1 ::
  -- | module
  ScriptModule ->
  -- | func
  String ->
  -- | inputs
  IValue ->
  -- | output
  IValue
runMethod1 module' func = unsafePerformIO . cast3 runMethod1' module' func
  where
    runMethod1' :: ScriptModule -> String -> RawIValue -> IO RawIValue
    runMethod1' = cast3 LibTorch.runMethod1

instance Parameterized ScriptModule where
  flattenParameters module' = map IndependentTensor $ getParameters module'
  _replaceParameters module' = do
    let len = length (getParameters module')
    ps' <- replicateM len nextParameter
    return $ updateParameters WithRequiredGrad module' (map toDependent ps')

trace ::
  -- | moduleName
  String ->
  -- | functionName
  String ->
  -- | function
  ([Tensor] -> IO [Tensor]) ->
  -- | inputs
  [Tensor] ->
  -- | output
  IO RawModule
trace moduleName functionName func = cast3 (\m f inps -> LibTorch.trace m f (trans func) inps) moduleName functionName
  where
    trans :: ([Tensor] -> IO [Tensor]) -> ForeignPtr TensorList -> IO (ForeignPtr TensorList)
    trans func inputs =
      uncast inputs $ \inputs' -> do
        ret <- func inputs'
        cast ret return

-- | This function generates torchscript-module from Parameterized-instance of hasktorch.
-- Usage is below.
-- -- >> let example_inputs = asTensor (4::Float)
-- -- >> init_parameters <- sample MonoSpec
-- -- >> mutableTorchscript <- traceWithParameters "MyModule"
-- --                            (\parameters [example_inputs'] -> return [(traced_function parameters example_inputs')])
-- --                            init_parameters
-- --                            [example_inputs]
-- -- >> immutableTorchscript <- toScriptModule mutableTorchscript
-- -- >> save immutableTorchscript "<your torchscript file>"
traceWithParameters ::
  Parameterized f =>
  -- | module name
  String ->
  -- | traced function
  (f -> [Tensor] -> IO [Tensor]) ->
  -- | initial parameters
  f ->
  -- | example inputs
  [Tensor] ->
  -- | torchscript module
  IO RawModule
traceWithParameters moduleName func parameterized_parameters inputs = do
  let parameters = map toDependent (flattenParameters parameterized_parameters)
      fromParams params = replaceParameters parameterized_parameters (map IndependentTensor params)
      plen = length parameters
      ilen = length inputs
  r <-
    trace
      moduleName
      "forwardWithParameters"
      ( \parametersAndInputs ->
          func
            (fromParams (take plen parametersAndInputs))
            (drop plen parametersAndInputs)
      )
      (parameters ++ inputs)
  forM_ (zip [0 ..] parameters) $ \(i, p) ->
    registerParameter r ("p" ++ show i) p False
  let args = intercalate ", " $ map (\i -> "i" ++ show i) [0 .. (ilen -1)]
      params = intercalate ", " $ map (\i -> "self.p" ++ show i) [0 .. (plen -1)]
  define r $
    "def forward(self, " ++ args ++ "):\n" ++ "    return self.forwardWithParameters(" ++ params ++ ", " ++ args ++ " )\n"
  return r

traceAsGraph ::
  -- | function
  ([Tensor] -> IO [Tensor]) ->
  -- | inputs
  [Tensor] ->
  -- | output
  IO Graph
traceAsGraph func = cast1 (LibTorch.traceAsGraph (trans func))
  where
    trans :: ([Tensor] -> IO [Tensor]) -> ForeignPtr TensorList -> IO (ForeignPtr TensorList)
    trans func inputs =
      uncast inputs $ \inputs' -> do
        ret <- func inputs'
        cast ret return

printGraph :: Graph -> IO String
printGraph = cast1 LibTorch.printGraph

-- | Output onnx file from graph. (really experimental implementation)
-- printOnnx uses export_onnx function of libtorch.
-- It outputs following error, because prim::Constant symbol using torchscript does not exist.
-- -- Exception: ONNX export failed: Couldn't export operator prim::Constant
-- -- Defined at:
-- --   Graph we tried to export:
-- --   graph(%0 : Float(),
-- --               %1 : Float()):
-- --     %2 : int = prim::Constant[value=1]()
-- --   %3 : Float() = aten::add(%0, %1, %2)
-- --   return (%3)
-- -- ; type: std::runtime_error
-- On the other hand, torch.onnx.export of python works.
-- onnx's symbol map is in python code.
-- https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py
--
-- If you need onnx-file, at first make torchscript by trace , then convert torchscript into onnx by python-code.
printOnnx :: Graph -> IO String
printOnnx = cast1 LibTorch.printOnnx

graphToJitGraph :: Graph -> IO JitGraph
graphToJitGraph (UnsafeGraph graph) =
  withForeignPtr graph $ \g0 -> Unmanaged.withJitGraph g0 $ \g -> do
    graphInputs <- toJitValue =<< Unmanaged.graphInputs g
    graphOutputs <- toJitValue =<< Unmanaged.graphOutputs g
    graphNodes <- toJitNode =<< Unmanaged.graphNodes g
    pure JitGraph {..}
  where
    toJitValue inputs =
      forM inputs $ \i -> do
        valueId <- cast1 Unmanaged.valueId i
        valueType <- cast0 (cast1 Unmanaged.valueType i :: IO (ForeignPtr ATen.StdString))
        pure JitValue {..}
    toJitNode nodes =
      forM nodes $ \n -> do
        nodeInputs <- toJitValue =<< Unmanaged.nodeInputs n
        nodeOutputs <- toJitValue =<< Unmanaged.nodeOutputs n
        nodeKind <- cast0 (cast1 Unmanaged.nodeKind n :: IO (ForeignPtr ATen.StdString))
        pure JitNode {..}

instance Castable [IValue] [RawIValue] where
  cast a f = forM a (`cast` return) >>= f
  uncast a f = forM a (`uncast` return) >>= f

instance Castable IValue RawIValue where
  cast IVNone f = newIValue >>= f
  cast (IVTensor (Unsafe v)) f = toIValue v >>= f
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
    cdoubles <- forM v (`cast` return) :: IO [CDouble]
    c10list <- cast cdoubles return :: IO (ForeignPtr (ATen.C10List CDouble))
    f =<< toIValue c10list
  cast (IVBoolList v) f = do
    cbools <- forM v (`cast` return) :: IO [CBool]
    c10list <- cast cbools return :: IO (ForeignPtr (ATen.C10List CBool))
    f =<< toIValue c10list
  cast (IVString v) f = do
    v' <- cast v return :: IO (ForeignPtr ATen.StdString)
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
    c10list <- cast rawIValues return :: IO (ForeignPtr (ATen.C10Dict '(ATen.IValue, ATen.IValue)))
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
      [ (iValue_isNone obj, f IVNone),
        (iValue_isTensor obj, fromIValue obj >>= f . IVTensor . Unsafe),
        (iValue_isDouble obj, fromIValue obj >>= f . IVDouble),
        (iValue_isInt obj, fromIValue obj >>= f . IVInt),
        (iValue_isBool obj, fromIValue obj >>= f . IVBool),
        ( iValue_isString obj,
          do
            v <- fromIValue obj :: IO (ForeignPtr ATen.StdString)
            str <- uncast v return :: IO String
            f (IVString str)
        ),
        ( iValue_isTensorList obj,
          do
            v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List ATen.Tensor))
            ts <- uncast v' return :: IO [Tensor]
            f (IVTensorList ts)
        ),
        ( iValue_isDoubleList obj,
          do
            v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List CDouble))
            cdoubles <- uncast v' return :: IO [CDouble]
            doubles <- forM cdoubles (`uncast` return) :: IO [Double]
            f (IVDoubleList doubles)
        ),
        ( iValue_isIntList obj,
          do
            v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List Int64))
            ts <- uncast v' return :: IO [Int64]
            f (IVIntList ts)
        ),
        ( iValue_isBoolList obj,
          do
            v' <- fromIValue obj :: IO (ForeignPtr (ATen.C10List CBool))
            cbools <- uncast v' return :: IO [CBool]
            bools <- forM cbools (`uncast` return) :: IO [Bool]
            f (IVBoolList bools)
        ),
        ( iValue_isTuple obj,
          do
            c10tuple <- fromIValue obj :: IO (ForeignPtr (ATen.C10Ptr ATen.IVTuple))
            rawIValues <- uncast c10tuple return :: IO [RawIValue]
            ts <- uncast rawIValues return :: IO [IValue]
            f (IVTuple ts)
        ),
        ( iValue_isList obj,
          do
            c10list <- fromIValue obj :: IO (ForeignPtr (ATen.C10List ATen.IValue))
            rawIValues <- uncast c10list return :: IO [RawIValue]
            ts <- uncast rawIValues return :: IO [IValue]
            f (IVGenericList ts)
        ),
        ( iValue_isGenericDict obj,
          do
            c10list <- fromIValue obj :: IO (ForeignPtr (ATen.C10Dict '(ATen.IValue, ATen.IValue)))
            rawIValues <- uncast c10list return :: IO [(RawIValue, RawIValue)]
            ts <- forM rawIValues $ \(a, b) -> do
              a' <- uncast a return
              b' <- uncast b return
              return (a', b')
            f (IVGenericDict ts)
        ),
        (iValue_isBlob obj, f IVBlob),
        (iValue_isFuture obj, f IVFuture),
        (iValue_isDevice obj, f IVDevice),
        (iValue_isObject obj, f IVObject),
        (iValue_isCapsule obj, f IVCapsule)
      ]
    where
      select [] = throwIO $ userError "Unsupported IValue"
      select ((cond, body) : xs) =
        cond >>= \case
          1 -> body
          _ -> select xs
