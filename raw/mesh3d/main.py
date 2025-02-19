from mesh3d import core
from datetime import datetime

start = datetime.now()
core.run()
end = datetime.now()
duration = end - start
minutes = duration.total_seconds() / 60
print("It took {:0.2f} minutes to process the covfefe out of the mesh(es).".format(minutes))
