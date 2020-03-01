# JupyterLab Example

This provides nix-shell environment with both JupyterLab and hasktorch.

This example should be run from the `experimental/jupyter/` directory.

First, run:

`nix-shell --option sandbox false --command "jupyer lab"`

The command outputs a url to connect to JupyterLab.
Then, open your browser with the url.

## References

```
https://gist.github.com/MMesch/45256a435c9f78158445214a56527f68
https://github.com/tweag/jupyterWith
```
