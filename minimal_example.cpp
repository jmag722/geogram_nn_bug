#include <vector>
#include <span>
#include <geogram/points/nn_search.h>

int main(int argc, char const *argv[]) {
    std::vector<double> coords = {
        0.84507479, 0.16097309,
        0.55774455, 0.36807994,
        0.21494196, 0.38582404,
        0.4281645,  0.61134311
    };
    std::vector<double> values = {
        0.73638587, 0.01528966, 0.25404091, 0.60414585
    };
    std::vector<double> query = {
        0.08373669, 0.99776368,
        0.83234612, 0.03677736
    };
    size_t dimension = 2;
    GEO::NearestNeighborSearch_var interp = GEO::NearestNeighborSearch::create(dimension, "BNN");
    interp->set_points(coords.size() / dimension, coords.data());

    std::vector<size_t> indices_found;
    size_t isize = query.size() / dimension;
    indices_found.reserve(isize);
    std::span<const double> myspan(query);

    for (size_t i = 0; i < query.size(); i += dimension) {
        auto subspan = myspan.subspan(i, dimension);
        indices_found.push_back(
            interp->get_nearest_neighbor(subspan.data())
        );
    }
    for (auto &&i : indices_found) {
        std::cout << "Index found with geogram: " << i << std::endl;
    }

    return 0;
}
