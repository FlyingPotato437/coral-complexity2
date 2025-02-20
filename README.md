# coral-complexity-metrics

[!CAUTION]
This package is currently under development.

## Installation

```
pip install .
```

## Usage

### Shading

```python
from coral_complexity_metrics import Shading

sh = Shading()
sh.load_3d_model("path/to/.ply/file")

# calculate shading metric
sh.calculate()
```

### Geometric Measures

```python
from coral_complexity_metrics import GeometricMeasures

gm = GeometricMeasures()
gm.load_3d_model("path/to/.ply/file")

# calculate geometric measures
gm.calculate()
```
