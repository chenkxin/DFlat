# DFlat
## Environment

### 1. Create an environment using environment.yml

```bash
conda env create -f environment.yml
```

### 2. Manual Installation Environment

**1. Create a new Conda environment**

```bash
conda create -n neural-optical python=3.8 tensorflow scikit-learn numpy tensorflow-probability matplotlib
```

---

**2. Activate new environment**

```bash
conda activate neural-optical
```

------

**3. Check environment**

Enter Python, try to import:

```python
import tensorflow as tf
import numpy as np
import sklearn
import tensorflow-probability
import matplotlib
```

If there are no error messages, it indicates that the environment configuration is essentially complete.

------

## Run code

```bash
 # Please run this command from the parent directory of dflat.
 python -m dflat.neural_optical_layer.core.trainer_models
```

------
