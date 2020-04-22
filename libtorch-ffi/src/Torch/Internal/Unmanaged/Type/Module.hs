
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module Torch.Internal.Unmanaged.Type.Module where

import Control.Exception.Safe (bracket)
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Unmanaged.Helper
import Torch.Internal.Class
import Data.IORef

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<torch/script.h>"
C.include "<torch/csrc/jit/serialization/export.h>"
C.include "<vector>"
C.include "<iostream>"

-- From libtorch/include/torch/csrc/jit/script/module.h

newModule
  :: Ptr StdString -> IO (Ptr Module)
newModule name =
  [C.throwBlock| torch::jit::script::Module* { return new torch::jit::script::Module(
     *$(std::string* name)
    );
  }|]

deleteModule :: Ptr Module -> IO ()
deleteModule object = [C.throwBlock| void { delete $(torch::jit::script::Module* object);}|]

instance CppObject Module where
  fromPtr ptr = newForeignPtr ptr (deleteModule ptr)

deleteJitGraph :: Ptr (SharedPtr JitGraph) -> IO ()
deleteJitGraph object = [C.throwBlock| void { delete $(std::shared_ptr<torch::jit::Graph>* object);}|]

instance CppObject (SharedPtr JitGraph) where
  fromPtr ptr = newForeignPtr ptr (deleteJitGraph ptr)

deleteJitNode :: Ptr JitNode -> IO ()
deleteJitNode object = [C.throwBlock| void { delete $(torch::jit::Node* object);}|]

instance CppObject JitNode where
  fromPtr ptr = newForeignPtr ptr (deleteJitNode ptr)

deleteJitValue :: Ptr JitValue -> IO ()
deleteJitValue object = [C.throwBlock| void { delete $(torch::jit::Value* object);}|]

instance CppObject JitValue where
  fromPtr ptr = newForeignPtr ptr (deleteJitValue ptr)

save :: Ptr Module -> FilePath -> IO ()
save obj file = withCString file $ \cfile -> [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->save($(char* cfile));
  }|]

load :: FilePath -> IO (Ptr Module)
load file = withCString file $ \cfile -> [C.throwBlock| torch::jit::script::Module* {
    return new torch::jit::script::Module(torch::jit::load($(char* cfile)));
  }|]

forward :: Ptr Module -> (Ptr (StdVector IValue)) -> IO (Ptr IValue)
forward obj inputs = [C.throwBlock| at::IValue* {
    return new at::IValue($(torch::jit::script::Module* obj)->forward(*$(std::vector<at::IValue>* inputs)));
  }|]

registerParameter :: Ptr Module -> Ptr StdString -> Ptr Tensor -> CBool -> IO ()
registerParameter obj name v is_buffer = [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->register_parameter(
      *$(std::string* name)
    , *$(at::Tensor* v)
    , $(bool is_buffer)
    );
  }|]

registerModule :: Ptr Module -> Ptr StdString -> Ptr Module -> IO ()
registerModule obj name v = [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->register_module(
      *$(std::string* name)
    , *$(torch::jit::script::Module* v)
    );
  }|]

train :: Ptr Module -> CBool -> IO ()
train obj on = [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->train(
      $(bool on)
    );
  }|]

runMethod :: Ptr Module -> Ptr StdString -> Ptr (C10List IValue) -> IO (Ptr IValue)
runMethod obj method_name args = [C.throwBlock| at::IValue* {
    return new at::IValue($(torch::jit::script::Module* obj)->run_method(
      *$(std::string* method_name)
    , *$(c10::List<at::IValue>* args)
    ));
  }|]

runMethod1 :: Ptr Module -> Ptr StdString -> Ptr IValue -> IO (Ptr IValue)
runMethod1 obj method_name args = [C.throwBlock| at::IValue* {
    return new at::IValue($(torch::jit::script::Module* obj)->run_method(
      *$(std::string* method_name)
    , *$(at::IValue* args)
    ));
  }|]

getParameters :: Ptr Module -> IO (Ptr TensorList)
getParameters obj = [C.throwBlock| std::vector<at::Tensor>* {
    std::vector<at::Tensor>* vec_parameters = new std::vector<at::Tensor>();
    auto parameters = $(torch::jit::script::Module* obj)->parameters();
    for(auto p : parameters) {
      vec_parameters->push_back(p);
    }
    return vec_parameters;
  }|]

setParameters :: Ptr Module -> Ptr TensorList -> IO ()
setParameters obj params = [C.throwBlock| void {
    auto module = $(torch::jit::script::Module* obj);
    auto parameters = module->named_parameters();
    auto vec = $(std::vector<at::Tensor>* params);
    int i=0; 
    for(auto p : parameters) {
      module->register_parameter(p.name,(*vec)[i],false);
    }
  }|]

toDevice :: Ptr Module -> DeviceType -> Int16 -> IO ()
toDevice obj device device_index = [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->to(torch::Device($(at::DeviceType device), $(int16_t device_index)));
  }|]

clone :: Ptr Module -> IO (Ptr Module)
clone obj = [C.throwBlock| torch::jit::script::Module* {
    return new torch::jit::script::Module($(torch::jit::script::Module* obj)->clone());
  }|]

define :: Ptr Module -> Ptr StdString -> IO ()
define obj src = [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->define(
      *$(std::string* src)
    );
  }|]

trace :: CString -> CString -> (Ptr TensorList -> IO (Ptr TensorList)) -> Ptr TensorList -> IO (Ptr Module)
trace moduleName functionName func inputs =
  bracket
    (callbackHelper $ \inputs' -> castPtr <$> func (castPtr inputs'))
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| torch::jit::script::Module* {
        torch::jit::script::Module self($(char* moduleName));
        auto vars_in = *$(std::vector<at::Tensor>* inputs);
        auto tfunc = $(void* (*funcPtr)(void*));
        typedef std::vector<at::Tensor>* (*Func)(std::vector<at::Tensor>*);
        auto func = (Func)tfunc;
        auto graph = torch::jit::tracer::trace(
          c10::fmap<c10::IValue>(vars_in),
          [&func](c10::Stack in) -> c10::Stack {
            std::vector<at::Tensor>* ivalue_inps = new std::vector<at::Tensor>(c10::fmap(in, [](const c10::IValue& v){
              return torch::autograd::Variable(v.toTensor());
            }));
            std::vector<at::Tensor> out = *(func(ivalue_inps));
            return c10::fmap<c10::IValue>(out);
          },
          [](const torch::autograd::Variable& var) { return "";}
        ).first->graph;
        auto v = graph->insertInput(0, "self");
        v->setType(self._ivalue()->type());
        const auto name = c10::QualifiedName(*self.type()->name(), $(char* functionName));
        auto fn2 = self._ivalue()->compilation_unit()->create_function(name,graph);
        self.type()->addMethod(fn2);
        return new torch::jit::script::Module(self);
      }|]

traceAsGraph :: (Ptr TensorList -> IO (Ptr TensorList)) -> Ptr TensorList -> IO (Ptr (SharedPtr JitGraph))
traceAsGraph func inputs =
  bracket
    (callbackHelper $ \inputs' -> castPtr <$> func (castPtr inputs'))
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| std::shared_ptr<torch::jit::Graph>* {
        torch::jit::script::Module self("MyModule");
        auto vars_in = *$(std::vector<at::Tensor>* inputs);
        auto tfunc = $(void* (*funcPtr)(void*));
        typedef std::vector<at::Tensor>* (*Func)(std::vector<at::Tensor>*);
        auto func = (Func)tfunc;
        auto graph = torch::jit::tracer::trace(
          c10::fmap<c10::IValue>(vars_in),
          [&func](c10::Stack in) -> c10::Stack {
            std::vector<at::Tensor>* ivalue_inps = new std::vector<at::Tensor>(c10::fmap(in, [](const c10::IValue& v){
              return torch::autograd::Variable(v.toTensor());
            }));
            std::vector<at::Tensor> out = *(func(ivalue_inps));
            return c10::fmap<c10::IValue>(out);
          },
          [](const torch::autograd::Variable& var) { return "";}
        ).first->graph;
        return new std::shared_ptr<torch::jit::Graph>(graph);
      }|]

withJitGraph :: Ptr (SharedPtr JitGraph) -> (Ptr JitGraph -> IO a) -> IO a
withJitGraph graph callback = do
  v <- [C.throwBlock| torch::jit::Graph* {
         return (*$(std::shared_ptr<torch::jit::Graph>* graph)).get();
       }|]
  callback v

graphOutputs :: Ptr JitGraph -> IO [Ptr (JitValue)]
graphOutputs graph = do
  nodes <- newIORef []
  let func v = do
        r <- readIORef nodes
        writeIORef nodes (v:r)
        return v
  bracket
    (callbackHelper $ \inputs' -> castPtr <$> func (castPtr inputs'))
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| void {
        auto tfunc = $(void* (*funcPtr)(void*));
        typedef torch::jit::Value* (*Func)(torch::jit::Value*);
        auto func = (Func)tfunc;
        for(auto i : (*$(torch::jit::Graph* graph)).outputs()){
          func(i);
        }
      }|]
  reverse <$> readIORef nodes

graphInputs :: Ptr JitGraph -> IO [Ptr (JitValue)]
graphInputs graph = do
  nodes <- newIORef []
  let func v = do
        r <- readIORef nodes
        writeIORef nodes (v:r)
        return v
  bracket
    (callbackHelper $ \inputs' -> castPtr <$> func (castPtr inputs'))
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| void {
        auto tfunc = $(void* (*funcPtr)(void*));
        typedef torch::jit::Value* (*Func)(torch::jit::Value*);
        auto func = (Func)tfunc;
        for(auto i : (*$(torch::jit::Graph* graph)).inputs()){
          func(i);
        }
      }|]
  reverse <$> readIORef nodes

graphNodes :: Ptr JitGraph -> IO [Ptr (JitNode)]
graphNodes graph = do
  nodes <- newIORef []
  let func v = do
        r <- readIORef nodes
        writeIORef nodes (v:r)
        return v
  bracket
    (callbackHelper $ \inputs' -> castPtr <$> func (castPtr inputs'))
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| void {
        auto tfunc = $(void* (*funcPtr)(void*));
        typedef torch::jit::Node* (*Func)(torch::jit::Node*);
        auto func = (Func)tfunc;
        for(auto i : (*$(torch::jit::Graph* graph)).block()->nodes()){
          func(i);
        }
      }|]
  reverse <$> readIORef nodes

nodeInputs :: Ptr JitNode -> IO [Ptr (JitValue)]
nodeInputs node = do
  values <- newIORef []
  let func v = do
        r <- readIORef values
        writeIORef values (v:r)
        return v
  bracket
    (callbackHelper $ \inputs' -> castPtr <$> func (castPtr inputs'))
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| void {
        auto tfunc = $(void* (*funcPtr)(void*));
        typedef torch::jit::Value* (*Func)(torch::jit::Value*);
        auto func = (Func)tfunc;
        for(auto i : (*$(torch::jit::Node* node)).inputs()){
          func(i);
        }
      }|]
  reverse <$> readIORef values

nodeOutputs :: Ptr JitNode -> IO [Ptr (JitValue)]
nodeOutputs node = do
  values <- newIORef []
  let func v = do
        r <- readIORef values
        writeIORef values (v:r)
        return v
  bracket
    (callbackHelper $ \inputs' -> castPtr <$> func (castPtr inputs'))
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| void {
        auto tfunc = $(void* (*funcPtr)(void*));
        typedef torch::jit::Value* (*Func)(torch::jit::Value*);
        auto func = (Func)tfunc;
        for(auto i : (*$(torch::jit::Node* node)).outputs()){
          func(i);
        }
      }|]
  reverse <$> readIORef values

nodeKind :: Ptr JitNode -> IO (Ptr StdString)
nodeKind node =
  [C.throwBlock| std::string* {
    return new std::string((*$(torch::jit::Node* node)).kind().toQualString());
  }|]

valueId :: Ptr JitValue -> IO CInt
valueId value =
  [C.throwBlock| int {
    return (*$(torch::jit::Value* value)).unique();
  }|]

valueType :: Ptr JitValue -> IO (Ptr StdString)
valueType node =
  [C.throwBlock| std::string* {
    return new std::string((*$(torch::jit::Value* node)).type()->str());
  }|]


printGraph :: Ptr (SharedPtr JitGraph) -> IO (Ptr StdString)
printGraph graph =
  [C.throwBlock| std::string* {
    return new std::string((**$(std::shared_ptr<torch::jit::Graph>* graph)).toString());
  }|]

printOnnx :: Ptr (SharedPtr JitGraph) -> IO (Ptr StdString)
printOnnx graph =
  [C.throwBlock| std::string* {
    auto graph_str = torch::jit::pretty_print_onnx(
        *$(std::shared_ptr<torch::jit::Graph>* graph),
        std::map<std::string, at::Tensor>{},
        9,
        false);
    return new std::string(graph_str);
  }|]

dumpToStr
  :: Ptr Module
  -> CBool
  -> CBool
  -> CBool
  -> CInt
  -> IO (Ptr StdString)
dumpToStr obj print_method_bodies print_attr_values print_param_values level =
  [C.throwBlock| std::string* {
    return new std::string($(torch::jit::script::Module* obj)->dump_to_str(
      $(bool print_method_bodies)
    , $(bool print_attr_values)
    , $(bool print_param_values)
    , $(int level)
    ));
  }|]
