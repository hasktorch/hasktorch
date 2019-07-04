
class Recurrent cell where

	run :: cell   -- state of the cell
	    -> Tensor -- input at current timestep
	    -> Tensor -- tensor containing current hidden state
	    -> (Tensor -- output tensor
	       ,Tensor) -- tensor containing next hidden state

	runOverTimesteps :: cell -- state of the cell
	                 -> Int  -- number of timesteps
	                 -> Tensor -- input sequence
	                 -> Tensor -- initial hidden state
	                 -> (Tensor -- output
	                 	,Tensor) -- final hidden state 