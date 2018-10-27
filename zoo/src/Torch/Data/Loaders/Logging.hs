module Torch.Data.Loaders.Logging where

mkLog level hdr msg = "["++level++"][" ++ hdr ++ "] " ++ msg
mkError = mkLog "ERROR"
mkInfo = mkLog "INFO"


