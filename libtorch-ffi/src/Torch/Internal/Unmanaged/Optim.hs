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
import qualified Language.C.Inline.Cpp.Unsafe as C
import qualified Language.C.Types as C
import Torch.Internal.Type
import Torch.Internal.Unmanaged.Helper

C.context $ C.cppCtx <> mempty {C.ctxTypesTable = typeTable}

C.include "<fstream>"
C.include "<vector>"
C.include "<tuple>"

C.include "hasktorch_profile.h"

C.include "<torch/types.h>"

C.include "<torch/optim.h>"

C.include "<torch/serialize.h>"

-- optimizerWithAdam
--   :: CDouble
--   -> CDouble
--   -> CDouble
--   -> CDouble
--   -> CDouble
--   -> CBool
--   -> Ptr TensorList
--   -> (Ptr TensorList -> IO (Ptr Tensor))
--   -> CInt
--   -> IO (Ptr TensorList)
-- optimizerWithAdam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad initParams loss numIter =
--   bracket
--     (callbackHelper loss')
--     freeHaskellFunPtr
--     $ \funcPtr ->
--       [C.throwBlock| std::vector<at::Tensor>* {
--         std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
--         std::vector<at::Tensor>* params = new std::vector<at::Tensor>();
--         auto tfunc = $(void* (*funcPtr)(void*));
--         for(int i=0;i<init_params->size();i++){
--           params->push_back((*init_params)[i].detach().set_requires_grad(true));
--         }
--         auto options = torch::optim::AdamOptions()
--           .lr($(double adamLr))
--           .betas(std::make_tuple($(double adamBetas0),$(double adamBetas1)))
--           .eps($(double adamEps))
--           .weight_decay($(double adamWeightDecay))
--           .amsgrad($(bool adamAmsgrad));
--         torch::optim::Adam optimizer(*params, options);
--         optimizer.zero_grad();
--         typedef at::Tensor* (*Func)(std::vector<at::Tensor>*);
--         auto func = (Func)tfunc;
--         for(int i=0;i<$(int numIter);i++){
--           auto loss = func(params);
--           loss->backward();
--           optimizer.step();
--         }
--         return params;
--       }|]
--   where
--     loss' :: Ptr () -> IO (Ptr ())
--     loss' params = castPtr <$> loss (castPtr params)


adagrad
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> Ptr TensorList
  -> IO (Ptr Optimizer)
adagrad lr lr_decay weight_decay initial_accumulator_value eps initParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
    std::vector<at::Tensor> params;
    for(int i=0;i<init_params->size();i++){
      params.push_back((*init_params)[i].detach().set_requires_grad(true));
    }
    auto options = torch::optim::AdagradOptions()
      .lr($(double lr))
      .lr_decay($(double lr_decay))
      .weight_decay($(double weight_decay))
      .initial_accumulator_value($(double initial_accumulator_value))
      .eps($(double eps));
    torch::optim::Adagrad* optimizer = new torch::optim::Adagrad(params, options);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]

rmsprop
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> Ptr TensorList
  -> IO (Ptr Optimizer)
rmsprop lr alpha eps weight_decay momentum centered initParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
    std::vector<at::Tensor> params;
    for(int i=0;i<init_params->size();i++){
      params.push_back((*init_params)[i].detach().set_requires_grad(true));
    }
    auto options = torch::optim::RMSpropOptions()
      .lr($(double lr))
      .alpha($(double alpha))
      .eps($(double eps))
      .weight_decay($(double weight_decay))
      .momentum($(double momentum))
      .centered($(bool centered));
    torch::optim::RMSprop* optimizer = new torch::optim::RMSprop(params, options);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]

sgd
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> Ptr TensorList
  -> IO (Ptr Optimizer)
sgd lr momentum dampening weight_decay nesterov initParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
    std::vector<at::Tensor> params;
    for(int i=0;i<init_params->size();i++){
      params.push_back((*init_params)[i].detach().set_requires_grad(true));
    }
    auto options = torch::optim::SGDOptions($(double lr))
      .momentum($(double momentum))
      .dampening($(double dampening))
      .weight_decay($(double weight_decay))
      .nesterov($(bool nesterov));
    torch::optim::SGD* optimizer = new torch::optim::SGD(params, options);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]

adam
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> Ptr TensorList
  -> IO (Ptr Optimizer)
adam adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad initParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
    std::vector<at::Tensor> params;
    for(int i=0;i<init_params->size();i++){
      params.push_back((*init_params)[i].detach().set_requires_grad(true));
    }
    auto options = torch::optim::AdamOptions()
      .lr($(double adamLr))
      .betas(std::make_tuple($(double adamBetas0),$(double adamBetas1)))
      .eps($(double adamEps))
      .weight_decay($(double adamWeightDecay))
      .amsgrad($(bool adamAmsgrad));
    torch::optim::Adam* optimizer = new torch::optim::Adam(params, options);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]

adamw
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> Ptr TensorList
  -> IO (Ptr Optimizer)
adamw adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad initParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
    std::vector<at::Tensor> params;
    for(int i=0;i<init_params->size();i++){
      params.push_back((*init_params)[i].detach().set_requires_grad(true));
    }
    auto options = torch::optim::AdamWOptions()
      .lr($(double adamLr))
      .betas(std::make_tuple($(double adamBetas0),$(double adamBetas1)))
      .eps($(double adamEps))
      .weight_decay($(double adamWeightDecay))
      .amsgrad($(bool adamAmsgrad));
    torch::optim::AdamW* optimizer = new torch::optim::AdamW(params, options);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]


lbfgs
  :: CDouble
  -> CInt
  -> CInt
  -> CDouble
  -> CDouble
  -> CInt
  -> Maybe (Ptr StdString)
  -> Ptr TensorList
  -> IO (Ptr Optimizer)
lbfgs lr max_iter max_eval tolerance_grad tolerance_change history_size Nothing initParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
    std::vector<at::Tensor> params;
    for(int i=0;i<init_params->size();i++){
      params.push_back((*init_params)[i].detach().set_requires_grad(true));
    }
    auto options = torch::optim::LBFGSOptions()
      .lr($(double lr))
      .max_iter($(int max_iter))
      .max_eval($(int max_eval))
      .tolerance_grad($(double tolerance_grad))
      .tolerance_change($(double tolerance_change))
      .history_size($(int history_size));
    torch::optim::LBFGS* optimizer = new torch::optim::LBFGS(params, options);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]
lbfgs lr max_iter max_eval tolerance_grad tolerance_change history_size (Just line_search_fn) initParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* init_params = $(std::vector<at::Tensor>* initParams);
    std::vector<at::Tensor> params;
    for(int i=0;i<init_params->size();i++){
      params.push_back((*init_params)[i].detach().set_requires_grad(true));
    }
    auto options = torch::optim::LBFGSOptions()
      .lr($(double lr))
      .max_iter($(int max_iter))
      .max_eval($(int max_eval))
      .tolerance_grad($(double tolerance_grad))
      .tolerance_change($(double tolerance_change))
      .history_size($(int history_size))
      .line_search_fn(*$(std::string* line_search_fn));
    torch::optim::LBFGS* optimizer = new torch::optim::LBFGS(params, options);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]

getParams :: Ptr Optimizer -> IO (Ptr TensorList)
getParams optimizer =
  [C.throwBlock| std::vector<at::Tensor>* {
    auto optimizer = $(torch::optim::Optimizer* optimizer);
    std::vector<at::Tensor>* result = new std::vector<at::Tensor>();
    for(auto& group : optimizer->param_groups()){
      for(auto& p : group.params()){
        result->push_back(p);
      }
    }
    return result;
  }|]

step :: Ptr Optimizer -> (Ptr TensorList -> IO (Ptr Tensor)) -> IO (Ptr Tensor)
step optimizer lossFunc =
  bracket
    (callbackHelper lossFunc')
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| at::Tensor* {
        auto tfunc = $(void* (*funcPtr)(void*));
        auto optimizer = $(torch::optim::Optimizer* optimizer);
        typedef at::Tensor* (*Func)(std::vector<at::Tensor>*);
        auto func = (Func)tfunc;
        auto v = optimizer->step([&]{
          optimizer->zero_grad();
          std::vector<at::Tensor> all_params;
          for(auto& group : optimizer->param_groups()){
            for(auto& p : group.params()){
              all_params.push_back(p);
            }
          }
          auto loss = func(&all_params);
          loss->backward();
          return *loss;
        });
        return new at::Tensor(v);
      }|]
  where
    lossFunc' :: Ptr () -> IO (Ptr ())
    lossFunc' params = castPtr <$> lossFunc (castPtr params)

stepWithGenerator :: Ptr Optimizer -> Ptr Generator -> (Ptr TensorList -> Ptr Generator -> IO (Ptr (StdTuple '(Tensor,Generator)))) -> IO (Ptr (StdTuple '(Tensor,Generator)))
stepWithGenerator optimizer generator lossFunc =
  bracket
    (callbackHelper2 lossFunc')
    freeHaskellFunPtr
    $ \funcPtr ->
      [C.throwBlock| std::tuple<at::Tensor,at::Generator>* {
        auto tfunc = $(void* (*funcPtr)(void*,void*));
        auto optimizer = $(torch::optim::Optimizer* optimizer);
        typedef std::tuple<at::Tensor,at::Generator>* (*Func)(std::vector<at::Tensor>*,at::Generator*);
        auto generator = $(at::Generator* generator)->clone();
        auto func = (Func)tfunc;
        auto v = optimizer->step([&]{
          optimizer->zero_grad();
          std::vector<at::Tensor> all_params;
          for(auto& group : optimizer->param_groups()){
            for(auto& p : group.params()){
              all_params.push_back(p);
            }
          }
          auto lossWithGenerator = func(&all_params,&generator);
          auto loss = std::get<0>(*lossWithGenerator);
          generator = std::get<1>(*lossWithGenerator);
          loss.backward();
          return loss;
        });
        return new std::tuple<at::Tensor,at::Generator>(std::make_tuple(v,generator));
      }|]
  where
    lossFunc' :: Ptr () -> Ptr () -> IO (Ptr ())
    lossFunc' params generator = castPtr <$> lossFunc (castPtr params) (castPtr generator)

-- After this function is called, params(TensorList) of optimizer is updated.
-- TensorList of output is the same as optimizer's params(TensorList).
unsafeStep :: Ptr Optimizer -> Ptr Tensor -> IO (Ptr TensorList)
unsafeStep optimizer loss =
  [C.throwBlock| std::vector<at::Tensor>* {
    auto optimizer = $(torch::optim::Optimizer* optimizer);
    auto loss = $(at::Tensor* loss);
    optimizer->zero_grad();
    loss->backward();
    optimizer->step();
    std::vector<at::Tensor>* result = new std::vector<at::Tensor>();
    for(auto& group : optimizer->param_groups()){
      for(auto& p : group.params()){
        result->push_back(p);
      }
    }
    return result;
  }|]

save :: Ptr Optimizer -> Ptr StdString -> IO ()
save optimizer filename =
  [C.throwBlock| void {
    std::ofstream output(*$(std::string* filename));
    torch::save(*$(torch::optim::Optimizer* optimizer),output);
  }|]

load :: Ptr Optimizer -> Ptr StdString -> IO ()
load optimizer filename =
  [C.throwBlock| void {
    std::ifstream input(*$(std::string* filename));
    torch::load(*$(torch::optim::Optimizer* optimizer),input);
  }|]

adamwWithParamGroups
  :: CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CDouble
  -> CBool
  -> Ptr TensorList
  -> Ptr TensorList
  -> IO (Ptr Optimizer)
adamwWithParamGroups adamLr adamBetas0 adamBetas1 adamEps adamWeightDecay adamAmsgrad decayParams noDecayParams =
  [C.throwBlock| torch::optim::Optimizer* {
    std::vector<at::Tensor>* decay_params = $(std::vector<at::Tensor>* decayParams);
    std::vector<at::Tensor>* no_decay_params = $(std::vector<at::Tensor>* noDecayParams);
    std::vector<at::Tensor> dp;
    for(int i=0;i<decay_params->size();i++){
      dp.push_back((*decay_params)[i].detach().set_requires_grad(true));
    }
    std::vector<at::Tensor> ndp;
    for(int i=0;i<no_decay_params->size();i++){
      ndp.push_back((*no_decay_params)[i].detach().set_requires_grad(true));
    }
    auto options_decay = torch::optim::AdamWOptions()
      .lr($(double adamLr))
      .betas(std::make_tuple($(double adamBetas0),$(double adamBetas1)))
      .eps($(double adamEps))
      .weight_decay($(double adamWeightDecay))
      .amsgrad($(bool adamAmsgrad));
    auto options_no_decay = torch::optim::AdamWOptions()
      .lr($(double adamLr))
      .betas(std::make_tuple($(double adamBetas0),$(double adamBetas1)))
      .eps($(double adamEps))
      .weight_decay(0.0)
      .amsgrad($(bool adamAmsgrad));
    std::vector<torch::optim::OptimizerParamGroup> param_groups;
    param_groups.emplace_back(torch::optim::OptimizerParamGroup(ndp, std::make_unique<torch::optim::AdamWOptions>(options_no_decay)));
    param_groups.emplace_back(torch::optim::OptimizerParamGroup(dp, std::make_unique<torch::optim::AdamWOptions>(options_decay)));
    torch::optim::AdamW* optimizer = new torch::optim::AdamW(param_groups);
    optimizer->zero_grad();
    return dynamic_cast<torch::optim::Optimizer*>(optimizer);
  }|]

getAllParams :: Ptr Optimizer -> IO (Ptr TensorList)
getAllParams optimizer =
  [C.throwBlock| std::vector<at::Tensor>* {
    auto optimizer = $(torch::optim::Optimizer* optimizer);
    std::vector<at::Tensor>* result = new std::vector<at::Tensor>();
    for(auto& group : optimizer->param_groups()){
      for(auto& p : group.params()){
        result->push_back(p);
      }
    }
    return result;
  }|]

stepOnly :: Ptr Optimizer -> IO ()
stepOnly optimizer =
  [C.throwBlock| void {
    $(torch::optim::Optimizer* optimizer)->step();
  }|]

zeroGrad :: Ptr Optimizer -> IO ()
zeroGrad optimizer =
  [C.throwBlock| void {
    $(torch::optim::Optimizer* optimizer)->zero_grad();
  }|]

setParamGrads :: Ptr Optimizer -> Ptr TensorList -> IO ()
setParamGrads optimizer grads =
  [C.throwBlock| void {
    auto optimizer = $(torch::optim::Optimizer* optimizer);
    auto grads = $(std::vector<at::Tensor>* grads);
    int idx = 0;
    for(auto& group : optimizer->param_groups()){
      for(auto& p : group.params()){
        p.mutable_grad() = (*grads)[idx];
        idx++;
      }
    }
  }|]

setLr :: Ptr Optimizer -> CDouble -> IO ()
setLr optimizer newLr =
  [C.throwBlock| void {
    auto optimizer = $(torch::optim::Optimizer* optimizer);
    for(auto& group : optimizer->param_groups()){
      auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
      options.lr($(double newLr));
    }
  }|]

setGroupLr :: Ptr Optimizer -> CInt -> CDouble -> IO ()
setGroupLr optimizer groupIdx lr =
  [C.throwBlock| void {
    auto optimizer = $(torch::optim::Optimizer* optimizer);
    auto& group = optimizer->param_groups()[(size_t)$(int groupIdx)];
    auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
    options.lr($(double lr));
  }|]
