{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}

module Torch.Internal.Unmanaged.Optim where

import Control.Exception.Safe (bracket)
import Foreign
import Foreign.C.String
import Foreign.C.Types
import Foreign.Ptr
import qualified Language.C.Inline.Context as C
import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Types as C
import Torch.Internal.Type
import Torch.Internal.Unmanaged.Helper

C.context $ C.cppCtx <> mempty {C.ctxTypesTable = typeTable}

C.include "<vector>"

C.include "<torch/types.h>"

C.include "<torch/optim.h>"


optimizerWithAdam
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> Ptr TensorList
  -> (Ptr TensorList -> IO (Ptr Tensor))
  -> CInt
  -> IO (Ptr TensorList)
optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad initParams loss numIter =
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
        auto options = torch::optim::AdamOptions()
          .lr($(double adamLr))
          .betas(std::make_tuple($(double adamBetas0),$(double adamBetas1)))
          .eps($(double adamEps))
          .weight_decay($(double adamWeightDecay))
          .amsgrad($(bool adamAmsgrad));
        torch::optim::Adam optimizer(*params, options);
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
