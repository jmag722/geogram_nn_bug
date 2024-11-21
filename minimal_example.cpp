#include <vector>
#include <geogram/points/nn_search.h>

int main(int argc, char const *argv[]) {
    std::vector<double> coords = {
        // .1, .8, // without this line, cout outputs `9` below. with it I get `11`
        0.63696169, 0.26978671,
        0.04097352, 0.01652764,
        0.81327024, 0.91275558,
        0.60663578, 0.72949656,
        0.54362499, 0.93507242,
    };
    // if I comment out Values (which isn't used) I get a different answer below (index 7!)
    std::vector<double> values = {
        // 0.5,
        0.81585355, 0.0027385, 0.85740428, 0.03358558, 0.72965545};
    std::vector<double> query = {0.17565562, 0.86317892};

    GEO::NearestNeighborSearch_var interp = GEO::NearestNeighborSearch::create(2, "BNN");
    interp->set_points(coords.size() / 2, coords.data());
    size_t found_index = interp->get_nearest_neighbor(query.data());
    std::cout << "Index found with geogram: " << found_index << std::endl;

    return 0;
}
