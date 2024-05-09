<h1 align='center'>Cognax</h1>

Build, fit, and sample from cognitive models with JAX + NumPyro.


Cognax currently has three submodules:

- `cognax.decisions`: tools for modeling decision making
- `cognax.joint`: tools for joint modeling (e.g. neural and behavioral data) 
- `cognax.experimental`: everything else

> [!WARNING]  
> This project is still in early development. The API is subject to change, and the code has not been thoroughly bug tested. 
 
## Installation

You can install the latest development version of Cognax directly from GitHub using the following command:

```bash
pip install git+https://github.com/compmem/cognax
```

> [!NOTE] 
> JAX is not listed as a dependency, but it must be installed separately before using Cognax. See the [JAX Docs](https://jax.readthedocs.io/en/latest/installation.html) for more information.
