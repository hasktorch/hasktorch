
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Managed.Native.Native14 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Native.Native14 as Unmanaged


special_logit_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_logit_t = cast1 Unmanaged.special_logit_t

special_logit_out_ttd
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
special_logit_out_ttd = cast3 Unmanaged.special_logit_out_ttd

special_logit_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_logit_out_tt = cast2 Unmanaged.special_logit_out_tt

special_polygamma_lt
  :: Int64
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_polygamma_lt = cast2 Unmanaged.special_polygamma_lt

special_polygamma_out_tlt
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_polygamma_out_tlt = cast3 Unmanaged.special_polygamma_out_tlt

special_logsumexp_tlb
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
special_logsumexp_tlb = cast3 Unmanaged.special_logsumexp_tlb

special_logsumexp_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
special_logsumexp_tl = cast2 Unmanaged.special_logsumexp_tl

special_logsumexp_out_ttlb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> CBool
  -> IO (ForeignPtr Tensor)
special_logsumexp_out_ttlb = cast4 Unmanaged.special_logsumexp_out_ttlb

special_logsumexp_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
special_logsumexp_out_ttl = cast3 Unmanaged.special_logsumexp_out_ttl

special_expit_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_expit_t = cast1 Unmanaged.special_expit_t

special_expit_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_expit_out_tt = cast2 Unmanaged.special_expit_out_tt

special_sinc_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_sinc_t = cast1 Unmanaged.special_sinc_t

special_sinc_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_sinc_out_tt = cast2 Unmanaged.special_sinc_out_tt

special_round_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
special_round_tl = cast2 Unmanaged.special_round_tl

special_round_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_round_t = cast1 Unmanaged.special_round_t

special_round_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
special_round_out_ttl = cast3 Unmanaged.special_round_out_ttl

special_round_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_round_out_tt = cast2 Unmanaged.special_round_out_tt

special_log1p_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_log1p_t = cast1 Unmanaged.special_log1p_t

special_log1p_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_log1p_out_tt = cast2 Unmanaged.special_log1p_out_tt

special_log_softmax_tls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
special_log_softmax_tls = cast3 Unmanaged.special_log_softmax_tls

special_log_softmax_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
special_log_softmax_tl = cast2 Unmanaged.special_log_softmax_tl

special_gammainc_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_gammainc_out_ttt = cast3 Unmanaged.special_gammainc_out_ttt

special_gammainc_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_gammainc_tt = cast2 Unmanaged.special_gammainc_tt

special_gammaincc_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_gammaincc_out_ttt = cast3 Unmanaged.special_gammaincc_out_ttt

special_gammaincc_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
special_gammaincc_tt = cast2 Unmanaged.special_gammaincc_tt

special_multigammaln_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
special_multigammaln_tl = cast2 Unmanaged.special_multigammaln_tl

special_multigammaln_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
special_multigammaln_out_ttl = cast3 Unmanaged.special_multigammaln_out_ttl

special_softmax_tls
  :: ForeignPtr Tensor
  -> Int64
  -> ScalarType
  -> IO (ForeignPtr Tensor)
special_softmax_tls = cast3 Unmanaged.special_softmax_tls

special_softmax_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
special_softmax_tl = cast2 Unmanaged.special_softmax_tl

fft_fft_tlls
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_fft_tlls = cast4 Unmanaged.fft_fft_tlls

fft_fft_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_fft_tll = cast3 Unmanaged.fft_fft_tll

fft_fft_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_fft_tl = cast2 Unmanaged.fft_fft_tl

fft_fft_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_fft_t = cast1 Unmanaged.fft_fft_t

fft_fft_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_fft_out_ttlls = cast5 Unmanaged.fft_fft_out_ttlls

fft_fft_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_fft_out_ttll = cast4 Unmanaged.fft_fft_out_ttll

fft_fft_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_fft_out_ttl = cast3 Unmanaged.fft_fft_out_ttl

fft_fft_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_fft_out_tt = cast2 Unmanaged.fft_fft_out_tt

fft_ifft_tlls
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ifft_tlls = cast4 Unmanaged.fft_ifft_tlls

fft_ifft_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ifft_tll = cast3 Unmanaged.fft_ifft_tll

fft_ifft_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ifft_tl = cast2 Unmanaged.fft_ifft_tl

fft_ifft_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ifft_t = cast1 Unmanaged.fft_ifft_t

fft_ifft_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ifft_out_ttlls = cast5 Unmanaged.fft_ifft_out_ttlls

fft_ifft_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ifft_out_ttll = cast4 Unmanaged.fft_ifft_out_ttll

fft_ifft_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ifft_out_ttl = cast3 Unmanaged.fft_ifft_out_ttl

fft_ifft_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ifft_out_tt = cast2 Unmanaged.fft_ifft_out_tt

fft_rfft_tlls
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_rfft_tlls = cast4 Unmanaged.fft_rfft_tlls

fft_rfft_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_rfft_tll = cast3 Unmanaged.fft_rfft_tll

fft_rfft_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_rfft_tl = cast2 Unmanaged.fft_rfft_tl

fft_rfft_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_rfft_t = cast1 Unmanaged.fft_rfft_t

fft_rfft_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_rfft_out_ttlls = cast5 Unmanaged.fft_rfft_out_ttlls

fft_rfft_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_rfft_out_ttll = cast4 Unmanaged.fft_rfft_out_ttll

fft_rfft_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_rfft_out_ttl = cast3 Unmanaged.fft_rfft_out_ttl

fft_rfft_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_rfft_out_tt = cast2 Unmanaged.fft_rfft_out_tt

fft_irfft_tlls
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_irfft_tlls = cast4 Unmanaged.fft_irfft_tlls

fft_irfft_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_irfft_tll = cast3 Unmanaged.fft_irfft_tll

fft_irfft_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_irfft_tl = cast2 Unmanaged.fft_irfft_tl

fft_irfft_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_irfft_t = cast1 Unmanaged.fft_irfft_t

fft_irfft_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_irfft_out_ttlls = cast5 Unmanaged.fft_irfft_out_ttlls

fft_irfft_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_irfft_out_ttll = cast4 Unmanaged.fft_irfft_out_ttll

fft_irfft_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_irfft_out_ttl = cast3 Unmanaged.fft_irfft_out_ttl

fft_irfft_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_irfft_out_tt = cast2 Unmanaged.fft_irfft_out_tt

fft_hfft_tlls
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_hfft_tlls = cast4 Unmanaged.fft_hfft_tlls

fft_hfft_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_hfft_tll = cast3 Unmanaged.fft_hfft_tll

fft_hfft_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_hfft_tl = cast2 Unmanaged.fft_hfft_tl

fft_hfft_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_hfft_t = cast1 Unmanaged.fft_hfft_t

fft_hfft_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_hfft_out_ttlls = cast5 Unmanaged.fft_hfft_out_ttlls

fft_hfft_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_hfft_out_ttll = cast4 Unmanaged.fft_hfft_out_ttll

fft_hfft_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_hfft_out_ttl = cast3 Unmanaged.fft_hfft_out_ttl

fft_hfft_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_hfft_out_tt = cast2 Unmanaged.fft_hfft_out_tt

fft_ihfft_tlls
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ihfft_tlls = cast4 Unmanaged.fft_ihfft_tlls

fft_ihfft_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ihfft_tll = cast3 Unmanaged.fft_ihfft_tll

fft_ihfft_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ihfft_tl = cast2 Unmanaged.fft_ihfft_tl

fft_ihfft_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ihfft_t = cast1 Unmanaged.fft_ihfft_t

fft_ihfft_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ihfft_out_ttlls = cast5 Unmanaged.fft_ihfft_out_ttlls

fft_ihfft_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ihfft_out_ttll = cast4 Unmanaged.fft_ihfft_out_ttll

fft_ihfft_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_ihfft_out_ttl = cast3 Unmanaged.fft_ihfft_out_ttl

fft_ihfft_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ihfft_out_tt = cast2 Unmanaged.fft_ihfft_out_tt

fft_fft2_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_fft2_tlls = cast4 Unmanaged.fft_fft2_tlls

fft_fft2_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fft2_tll = cast3 Unmanaged.fft_fft2_tll

fft_fft2_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fft2_tl = cast2 Unmanaged.fft_fft2_tl

fft_fft2_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_fft2_t = cast1 Unmanaged.fft_fft2_t

fft_fft2_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_fft2_out_ttlls = cast5 Unmanaged.fft_fft2_out_ttlls

fft_fft2_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fft2_out_ttll = cast4 Unmanaged.fft_fft2_out_ttll

fft_fft2_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fft2_out_ttl = cast3 Unmanaged.fft_fft2_out_ttl

fft_fft2_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_fft2_out_tt = cast2 Unmanaged.fft_fft2_out_tt

fft_ifft2_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ifft2_tlls = cast4 Unmanaged.fft_ifft2_tlls

fft_ifft2_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifft2_tll = cast3 Unmanaged.fft_ifft2_tll

fft_ifft2_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifft2_tl = cast2 Unmanaged.fft_ifft2_tl

fft_ifft2_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ifft2_t = cast1 Unmanaged.fft_ifft2_t

fft_ifft2_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ifft2_out_ttlls = cast5 Unmanaged.fft_ifft2_out_ttlls

fft_ifft2_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifft2_out_ttll = cast4 Unmanaged.fft_ifft2_out_ttll

fft_ifft2_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifft2_out_ttl = cast3 Unmanaged.fft_ifft2_out_ttl

fft_ifft2_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ifft2_out_tt = cast2 Unmanaged.fft_ifft2_out_tt

fft_rfft2_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_rfft2_tlls = cast4 Unmanaged.fft_rfft2_tlls

fft_rfft2_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfft2_tll = cast3 Unmanaged.fft_rfft2_tll

fft_rfft2_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfft2_tl = cast2 Unmanaged.fft_rfft2_tl

fft_rfft2_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_rfft2_t = cast1 Unmanaged.fft_rfft2_t

fft_rfft2_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_rfft2_out_ttlls = cast5 Unmanaged.fft_rfft2_out_ttlls

fft_rfft2_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfft2_out_ttll = cast4 Unmanaged.fft_rfft2_out_ttll

fft_rfft2_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfft2_out_ttl = cast3 Unmanaged.fft_rfft2_out_ttl

fft_rfft2_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_rfft2_out_tt = cast2 Unmanaged.fft_rfft2_out_tt

fft_irfft2_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_irfft2_tlls = cast4 Unmanaged.fft_irfft2_tlls

fft_irfft2_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfft2_tll = cast3 Unmanaged.fft_irfft2_tll

fft_irfft2_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfft2_tl = cast2 Unmanaged.fft_irfft2_tl

fft_irfft2_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_irfft2_t = cast1 Unmanaged.fft_irfft2_t

fft_irfft2_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_irfft2_out_ttlls = cast5 Unmanaged.fft_irfft2_out_ttlls

fft_irfft2_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfft2_out_ttll = cast4 Unmanaged.fft_irfft2_out_ttll

fft_irfft2_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfft2_out_ttl = cast3 Unmanaged.fft_irfft2_out_ttl

fft_irfft2_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_irfft2_out_tt = cast2 Unmanaged.fft_irfft2_out_tt

fft_hfft2_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_hfft2_tlls = cast4 Unmanaged.fft_hfft2_tlls

fft_hfft2_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfft2_tll = cast3 Unmanaged.fft_hfft2_tll

fft_hfft2_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfft2_tl = cast2 Unmanaged.fft_hfft2_tl

fft_hfft2_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_hfft2_t = cast1 Unmanaged.fft_hfft2_t

fft_hfft2_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_hfft2_out_ttlls = cast5 Unmanaged.fft_hfft2_out_ttlls

fft_hfft2_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfft2_out_ttll = cast4 Unmanaged.fft_hfft2_out_ttll

fft_hfft2_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfft2_out_ttl = cast3 Unmanaged.fft_hfft2_out_ttl

fft_hfft2_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_hfft2_out_tt = cast2 Unmanaged.fft_hfft2_out_tt

fft_ihfft2_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ihfft2_tlls = cast4 Unmanaged.fft_ihfft2_tlls

fft_ihfft2_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfft2_tll = cast3 Unmanaged.fft_ihfft2_tll

fft_ihfft2_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfft2_tl = cast2 Unmanaged.fft_ihfft2_tl

fft_ihfft2_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ihfft2_t = cast1 Unmanaged.fft_ihfft2_t

fft_ihfft2_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ihfft2_out_ttlls = cast5 Unmanaged.fft_ihfft2_out_ttlls

fft_ihfft2_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfft2_out_ttll = cast4 Unmanaged.fft_ihfft2_out_ttll

fft_ihfft2_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfft2_out_ttl = cast3 Unmanaged.fft_ihfft2_out_ttl

fft_ihfft2_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ihfft2_out_tt = cast2 Unmanaged.fft_ihfft2_out_tt

fft_fftn_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_fftn_tlls = cast4 Unmanaged.fft_fftn_tlls

fft_fftn_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fftn_tll = cast3 Unmanaged.fft_fftn_tll

fft_fftn_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fftn_tl = cast2 Unmanaged.fft_fftn_tl

fft_fftn_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_fftn_t = cast1 Unmanaged.fft_fftn_t

fft_fftn_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_fftn_out_ttlls = cast5 Unmanaged.fft_fftn_out_ttlls

fft_fftn_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fftn_out_ttll = cast4 Unmanaged.fft_fftn_out_ttll

fft_fftn_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fftn_out_ttl = cast3 Unmanaged.fft_fftn_out_ttl

fft_fftn_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_fftn_out_tt = cast2 Unmanaged.fft_fftn_out_tt

fft_ifftn_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ifftn_tlls = cast4 Unmanaged.fft_ifftn_tlls

fft_ifftn_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifftn_tll = cast3 Unmanaged.fft_ifftn_tll

fft_ifftn_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifftn_tl = cast2 Unmanaged.fft_ifftn_tl

fft_ifftn_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ifftn_t = cast1 Unmanaged.fft_ifftn_t

fft_ifftn_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ifftn_out_ttlls = cast5 Unmanaged.fft_ifftn_out_ttlls

fft_ifftn_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifftn_out_ttll = cast4 Unmanaged.fft_ifftn_out_ttll

fft_ifftn_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifftn_out_ttl = cast3 Unmanaged.fft_ifftn_out_ttl

fft_ifftn_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ifftn_out_tt = cast2 Unmanaged.fft_ifftn_out_tt

fft_rfftn_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_rfftn_tlls = cast4 Unmanaged.fft_rfftn_tlls

fft_rfftn_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfftn_tll = cast3 Unmanaged.fft_rfftn_tll

fft_rfftn_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfftn_tl = cast2 Unmanaged.fft_rfftn_tl

fft_rfftn_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_rfftn_t = cast1 Unmanaged.fft_rfftn_t

fft_rfftn_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_rfftn_out_ttlls = cast5 Unmanaged.fft_rfftn_out_ttlls

fft_rfftn_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfftn_out_ttll = cast4 Unmanaged.fft_rfftn_out_ttll

fft_rfftn_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_rfftn_out_ttl = cast3 Unmanaged.fft_rfftn_out_ttl

fft_rfftn_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_rfftn_out_tt = cast2 Unmanaged.fft_rfftn_out_tt

fft_irfftn_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_irfftn_tlls = cast4 Unmanaged.fft_irfftn_tlls

fft_irfftn_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfftn_tll = cast3 Unmanaged.fft_irfftn_tll

fft_irfftn_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfftn_tl = cast2 Unmanaged.fft_irfftn_tl

fft_irfftn_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_irfftn_t = cast1 Unmanaged.fft_irfftn_t

fft_irfftn_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_irfftn_out_ttlls = cast5 Unmanaged.fft_irfftn_out_ttlls

fft_irfftn_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfftn_out_ttll = cast4 Unmanaged.fft_irfftn_out_ttll

fft_irfftn_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_irfftn_out_ttl = cast3 Unmanaged.fft_irfftn_out_ttl

fft_irfftn_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_irfftn_out_tt = cast2 Unmanaged.fft_irfftn_out_tt

fft_hfftn_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_hfftn_tlls = cast4 Unmanaged.fft_hfftn_tlls

fft_hfftn_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfftn_tll = cast3 Unmanaged.fft_hfftn_tll

fft_hfftn_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfftn_tl = cast2 Unmanaged.fft_hfftn_tl

fft_hfftn_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_hfftn_t = cast1 Unmanaged.fft_hfftn_t

fft_hfftn_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_hfftn_out_ttlls = cast5 Unmanaged.fft_hfftn_out_ttlls

fft_hfftn_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfftn_out_ttll = cast4 Unmanaged.fft_hfftn_out_ttll

fft_hfftn_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_hfftn_out_ttl = cast3 Unmanaged.fft_hfftn_out_ttl

fft_hfftn_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_hfftn_out_tt = cast2 Unmanaged.fft_hfftn_out_tt

fft_ihfftn_tlls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ihfftn_tlls = cast4 Unmanaged.fft_ihfftn_tlls

fft_ihfftn_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfftn_tll = cast3 Unmanaged.fft_ihfftn_tll

fft_ihfftn_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfftn_tl = cast2 Unmanaged.fft_ihfftn_tl

fft_ihfftn_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ihfftn_t = cast1 Unmanaged.fft_ihfftn_t

fft_ihfftn_out_ttlls
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr StdString
  -> IO (ForeignPtr Tensor)
fft_ihfftn_out_ttlls = cast5 Unmanaged.fft_ihfftn_out_ttlls

fft_ihfftn_out_ttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfftn_out_ttll = cast4 Unmanaged.fft_ihfftn_out_ttll

fft_ihfftn_out_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ihfftn_out_ttl = cast3 Unmanaged.fft_ihfftn_out_ttl

fft_ihfftn_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ihfftn_out_tt = cast2 Unmanaged.fft_ihfftn_out_tt

fft_fftfreq_ldo
  :: Int64
  -> CDouble
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
fft_fftfreq_ldo = cast3 Unmanaged.fft_fftfreq_ldo

fft_fftfreq_ld
  :: Int64
  -> CDouble
  -> IO (ForeignPtr Tensor)
fft_fftfreq_ld = cast2 Unmanaged.fft_fftfreq_ld

fft_fftfreq_l
  :: Int64
  -> IO (ForeignPtr Tensor)
fft_fftfreq_l = cast1 Unmanaged.fft_fftfreq_l

fft_fftfreq_out_tld
  :: ForeignPtr Tensor
  -> Int64
  -> CDouble
  -> IO (ForeignPtr Tensor)
fft_fftfreq_out_tld = cast3 Unmanaged.fft_fftfreq_out_tld

fft_fftfreq_out_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_fftfreq_out_tl = cast2 Unmanaged.fft_fftfreq_out_tl

fft_rfftfreq_ldo
  :: Int64
  -> CDouble
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
fft_rfftfreq_ldo = cast3 Unmanaged.fft_rfftfreq_ldo

fft_rfftfreq_ld
  :: Int64
  -> CDouble
  -> IO (ForeignPtr Tensor)
fft_rfftfreq_ld = cast2 Unmanaged.fft_rfftfreq_ld

fft_rfftfreq_l
  :: Int64
  -> IO (ForeignPtr Tensor)
fft_rfftfreq_l = cast1 Unmanaged.fft_rfftfreq_l

fft_rfftfreq_out_tld
  :: ForeignPtr Tensor
  -> Int64
  -> CDouble
  -> IO (ForeignPtr Tensor)
fft_rfftfreq_out_tld = cast3 Unmanaged.fft_rfftfreq_out_tld

fft_rfftfreq_out_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
fft_rfftfreq_out_tl = cast2 Unmanaged.fft_rfftfreq_out_tl

fft_fftshift_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_fftshift_tl = cast2 Unmanaged.fft_fftshift_tl

fft_fftshift_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_fftshift_t = cast1 Unmanaged.fft_fftshift_t

fft_ifftshift_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
fft_ifftshift_tl = cast2 Unmanaged.fft_ifftshift_tl

fft_ifftshift_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
fft_ifftshift_t = cast1 Unmanaged.fft_ifftshift_t

linalg_cholesky_ex_tbb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_cholesky_ex_tbb = cast3 Unmanaged.linalg_cholesky_ex_tbb

linalg_cholesky_ex_tb
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_cholesky_ex_tb = cast2 Unmanaged.linalg_cholesky_ex_tb

linalg_cholesky_ex_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_cholesky_ex_t = cast1 Unmanaged.linalg_cholesky_ex_t

linalg_cholesky_ex_out_tttbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_cholesky_ex_out_tttbb = cast5 Unmanaged.linalg_cholesky_ex_out_tttbb

linalg_cholesky_ex_out_tttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_cholesky_ex_out_tttb = cast4 Unmanaged.linalg_cholesky_ex_out_tttb

linalg_cholesky_ex_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_cholesky_ex_out_ttt = cast3 Unmanaged.linalg_cholesky_ex_out_ttt

linalg_cholesky_tb
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
linalg_cholesky_tb = cast2 Unmanaged.linalg_cholesky_tb

linalg_cholesky_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linalg_cholesky_t = cast1 Unmanaged.linalg_cholesky_t

linalg_cholesky_out_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
linalg_cholesky_out_ttb = cast3 Unmanaged.linalg_cholesky_out_ttb

linalg_cholesky_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linalg_cholesky_out_tt = cast2 Unmanaged.linalg_cholesky_out_tt

linalg_cross_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
linalg_cross_ttl = cast3 Unmanaged.linalg_cross_ttl

linalg_cross_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linalg_cross_tt = cast2 Unmanaged.linalg_cross_tt

linalg_cross_out_tttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
linalg_cross_out_tttl = cast4 Unmanaged.linalg_cross_out_tttl

linalg_cross_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linalg_cross_out_ttt = cast3 Unmanaged.linalg_cross_out_ttt

linalg_lu_factor_tb
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_lu_factor_tb = cast2 Unmanaged.linalg_lu_factor_tb

linalg_lu_factor_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_lu_factor_t = cast1 Unmanaged.linalg_lu_factor_t

linalg_lu_factor_out_tttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_lu_factor_out_tttb = cast4 Unmanaged.linalg_lu_factor_out_tttb

linalg_lu_factor_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor)))
linalg_lu_factor_out_ttt = cast3 Unmanaged.linalg_lu_factor_out_ttt

linalg_lu_factor_ex_tbb
  :: ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
linalg_lu_factor_ex_tbb = cast3 Unmanaged.linalg_lu_factor_ex_tbb

linalg_lu_factor_ex_tb
  :: ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
linalg_lu_factor_ex_tb = cast2 Unmanaged.linalg_lu_factor_ex_tb

linalg_lu_factor_ex_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
linalg_lu_factor_ex_t = cast1 Unmanaged.linalg_lu_factor_ex_t

linalg_lu_factor_ex_out_ttttbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
linalg_lu_factor_ex_out_ttttbb = cast6 Unmanaged.linalg_lu_factor_ex_out_ttttbb

linalg_lu_factor_ex_out_ttttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
linalg_lu_factor_ex_out_ttttb = cast5 Unmanaged.linalg_lu_factor_ex_out_ttttb

linalg_lu_factor_ex_out_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
linalg_lu_factor_ex_out_tttt = cast4 Unmanaged.linalg_lu_factor_ex_out_tttt

linalg_det_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linalg_det_t = cast1 Unmanaged.linalg_det_t

linalg_det_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
linalg_det_out_tt = cast2 Unmanaged.linalg_det_out_tt

det_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
det_t = cast1 Unmanaged.det_t

