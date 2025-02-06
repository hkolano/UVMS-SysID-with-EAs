# UVMS-SysID-with-EAs




## Set up Environment

Set up julia. Set julia to the correct version.

```
curl -fsSL https://install.julialang.org | sh
. /home/gonzaeve/.bashrc
juliaup add 1.8.2
juliaup default 1.8.2
```

Set up python with the correct version and libraries.

```
conda create --name test4-py python=3.13.1
conda activate test4-py
pip install deap==1.4.2
pip install julia==0.6.2
pip install numpy==2.2.2
pip install PyYAML==6.0.2
pip install setuptools==75.8.0
pip install wheel==0.44.0
python
>>> import julia
>>> julia.install(julia='/home/gonzaeve/d.juliaup/bin/julia')
```


Set up PyCall on julia.

```
julia
>>> ENV["PYTHON"] = "/home/gonzaeve/miniconda3/envs/test4-py/bin/python"
>>> using Pkg
>>> Pkg.add("PyCall")
```
