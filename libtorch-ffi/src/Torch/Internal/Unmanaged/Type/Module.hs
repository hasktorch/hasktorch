
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
import Torch.Internal.Class

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<torch/script.h>"
C.include "<vector>"

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

register_parameter :: Ptr Module -> Ptr StdString -> Ptr Tensor -> CBool -> IO ()
register_parameter obj name v is_buffer = [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->register_parameter(
      *$(std::string* name)
    , *$(at::Tensor* v)
    , $(bool is_buffer)
    );
  }|]

register_module :: Ptr Module -> Ptr StdString -> Ptr Module -> IO ()
register_module obj name v = [C.throwBlock| void {
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

run_method :: Ptr Module -> Ptr StdString -> Ptr (C10List IValue) -> IO (Ptr IValue)
run_method obj method_name args = [C.throwBlock| at::IValue* {
    return new at::IValue($(torch::jit::script::Module* obj)->run_method(
      *$(std::string* method_name)
    , *$(c10::List<at::IValue>* args)
    ));
  }|]

run_method1 :: Ptr Module -> Ptr StdString -> Ptr IValue -> IO (Ptr IValue)
run_method1 obj method_name args = [C.throwBlock| at::IValue* {
    return new at::IValue($(torch::jit::script::Module* obj)->run_method(
      *$(std::string* method_name)
    , *$(at::IValue* args)
    ));
  }|]

define :: Ptr Module -> Ptr StdString -> IO ()
define obj src = [C.throwBlock| void {
    $(torch::jit::script::Module* obj)->define(
      *$(std::string* src)
    );
  }|]

trace :: (Ptr TensorList -> IO (Ptr TensorList)) -> Ptr TensorList -> IO (Ptr Module)
trace func inputs = do
  funcPtr <- $(C.mkFunPtr [t| Ptr TensorList -> IO (Ptr TensorList) |]) $ \inputs' -> func inputs'
  [C.throwBlock| torch::jit::script::Module* {
    torch::jit::script::Module self("M");
    auto vars_in = *$(std::vector<at::Tensor>* inputs);
    auto func = $(std::vector<at::Tensor>* (*funcPtr)(std::vector<at::Tensor>*));
    auto graph = torch::jit::tracer::trace(
      c10::fmap<c10::IValue>(vars_in),
      [&func](c10::Stack in) -> c10::Stack {
        auto ivalue_inps = c10::fmap(in, [](const c10::IValue& v){
          return torch::autograd::Variable(v.toTensor());
        });
        return c10::fmap<c10::IValue>(*(func(&ivalue_inps)));
      },
      [](const torch::autograd::Variable& var) { return "";}
    ).first->graph;
    auto v = graph->insertInput(0, "self");
    v->setType(self._ivalue()->type());
    const auto name = c10::QualifiedName(*self.type()->name(), "forward");
    auto fn2 = self._ivalue()->compilation_unit()->create_function(name,graph);
    self.type()->addMethod(fn2);
    return new torch::jit::script::Module(self);
  }|]
