{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Unmanaged.Optim where

import Foreign.Ptr
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C

import Torch.Internal.Type
import Torch.Internal.Unmanaged.Helper

import Control.Exception.Safe (bracket)
import Foreign.C.String
import Foreign.C.Types
import Foreign


C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<torch/types.h>"
C.include "<torch/optim.h>"

data AdamParams = AdamParams

optimizerWithAdam
  :: AdamParams
  -> Ptr TensorList
  -> (Ptr TensorList -> IO (Ptr Tensor))
  -> CInt
  -> IO (Ptr TensorList)
optimizerWithAdam optimizerParams initParams loss numIter =
  bracket
    (callbackHelper loss')
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| std::vector<at::Tensor>* {
        std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
        std::vector<at::Tensor>* params = new std::vector<at::Tensor>();
        auto tfunc = $(void* (*funcPtr)(void*));
        for(int i=0;i<init_params->size();i++){
          params->push_back((*init_params)[i].detach().set_requires_grad(true));
        }
        torch::optim::Adam optimizer(*params, torch::optim::AdamOptions(2e-4).weight_decay(1e-6));
        typedef at::Tensor* (*Func)(std::vector<at::Tensor>*);
        auto func = (Func)tfunc;
        for(int i=0;i<$(int numIter);i++){
          optimizer.step([&]{
            return *(func(params));
          });
        }
        return params;
      }|]
  where
    loss' :: Ptr () -> IO (Ptr ())
    loss' params = castPtr <$> loss (castPtr params)
    
