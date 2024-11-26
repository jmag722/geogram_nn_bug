import numpy as np
import scipy.interpolate as si
import deli as dm

coords = np.array([
    [0.84507479, 0.16097309],
    [0.55774455, 0.36807994],
    [0.21494196, 0.38582404],
    [0.4281645,  0.61134311]
])
values = np.array([
    0.73638587, 0.01528966, 0.25404091, 0.60414585
])
query = np.array([
    [0.08373669, 0.99776368],
    [0.83234612, 0.03677736]
])

pyinterp = si.NearestNDInterpolator(coords, values)
queried_value = pyinterp(query)

nn_distances, queried_index = pyinterp.tree.query(query, k=1)
queried_value_2 = values[queried_index]

my_indices = np.asarray([np.argmin(np.linalg.norm(q - coords, axis=1)) for q in query])
my_distances = np.asarray([np.min(np.linalg.norm(q - coords, axis=1)) for q in query])

np.testing.assert_equal(queried_value, queried_value_2)
np.testing.assert_equal(my_indices, queried_index)

print(f"Python Indices: {queried_index}")
print(f"Python neighbors: {queried_value}")

dim=2
cinterp = dm.NNInterpolator(dim, coords.flatten(), values)
orig_coords = cinterp.get_coords()
print(orig_coords)
cindices = np.asarray(cinterp.find_nn(query.flatten()))
cnewvals2 = values[cindices]
cnewvals = np.asarray(cinterp.get_values(cindices))
np.testing.assert_equal(queried_index, cindices)
np.testing.assert_equal(cnewvals, cnewvals2)
print(f"C++ Indices: {cindices}")
print(f"C++ neighbors: {cnewvals}")
