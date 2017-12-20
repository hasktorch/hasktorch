Dependencies to external repositories go here.

Currently pytorch is included as a submodule as the most recent changes to
`aten` and the `TH` core set of functions are maintained in the pytorch repo.

`build-aten.sh` is a script that builds the aten library within the pytorch
repo, including the shared `TH` library.
