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

sd = Shading()
sd.load_3d_model("path/to/.ply/file")

# calculate shading metric
sd.calculate()
```