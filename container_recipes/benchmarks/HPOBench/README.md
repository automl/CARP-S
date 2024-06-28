# Information about running HPOBench in a Container
Due to the setup of HPOBench, which uses containers itself, it does not run well within a container, so we do not offer
a recipe for it (old package versions are fixed if wanting to run HPOBench locally, which does not allow increasing the
python version etc. in the main carp-s environment/ container if this would be done. They cannot be increased easily
since old pickled surrogates need to be loaded in that environment in many cases). 

