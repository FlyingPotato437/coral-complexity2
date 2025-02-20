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
sh.load_mesh("path/to/.ply/file")

# calculate shading metric
sh.calculate()
```

### Geometric Measures

```python
from coral_complexity_metrics import GeometricMeasures

gm = GeometricMeasures()
gm.load_mesh("path/to/.ply/file")

# calculate geometric measures
gm.calculate()
```
