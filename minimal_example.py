import numpy as np
import scipy.interpolate as si

coords = np.array([
    # [.1, 0.8], # NOTE Uncommented, changes answer below to index 0. Commented out, answer is 4
    [0.63696169, 0.26978671],
    [0.04097352, 0.01652764],
    [0.81327024, 0.91275558],
    [0.60663578, 0.72949656],
    [0.54362499, 0.93507242]
])
values = np.array([
    # 0.5, # NOTE toggle on to be consistent with above
    0.81585355, 0.0027385, 0.85740428, 0.03358558, 0.72965545])
query = np.array([0.17565562, 0.86317892])

pyinterp = si.NearestNDInterpolator(coords, values)
queried_value = pyinterp(query)

nn_distances, queried_index = pyinterp.tree.query(query, k=1)
queried_value_2 = values[queried_index]

my_distances = np.linalg.norm(query - coords, axis=1)

np.testing.assert_equal(queried_value, queried_value_2)
np.testing.assert_equal(np.argmin(my_distances), queried_index)

print(f"Index found by Python: {queried_index}")
