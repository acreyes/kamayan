#include "grid/pybind/grid_py11.hpp"

#include <pybind11/attr.h>
#include <pybind11/detail/common.h>
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <set>
#include <string>
#include <vector>

#include <pack/make_pack_descriptor.hpp>
#include <pack/pack_descriptor.hpp>
#include <pack/sparse_pack.hpp>

#include "coordinates/coordinates.hpp"
#include "grid/grid_types.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/sparse_pack_base.hpp"

namespace kamayan {

struct SparsePack_py {
  SparsePack_py(MeshBlock *mb, const std::vector<std::string> vars) {
    auto pkg = mb->resolved_packages.get();
    auto desc = parthenon::MakePackDescriptor(pkg, vars, {}, {PDOpt::WithFluxes});
    pack = desc.GetPack(mb->meshblock_data.Get().get());
    map = desc.GetMap();
  }

  SparsePack_py(MeshData *md, const std::vector<std::string> vars,
                const std::set<parthenon::PDOpt> &options) {
    auto desc = parthenon::MakePackDescriptor(
        md->GetMeshPointer()->resolved_packages.get(), vars, {}, options);
    pack = desc.GetPack(md);
    map = desc.GetMap();
  }

  parthenon::ParArray3D<Real> GetParArray3D(const int &block, const std::string &var,
                                            const parthenon::TopologicalElement &te,
                                            const int &comp = 0) {
    return pack(block, te, map[var] + comp);
  }

  parthenon::Coordinates_t GetCoordinates(const int b) const {
    return pack.GetCoordinates(b);
  }

 private:
  parthenon::SparsePack<> pack;
  parthenon::SparsePackIdxMap map;
};

void sparse_pack_py(pybind11::module_ &m) {
  using PA3 = parthenon::ParArray3D<Real>;
  pybind11::classh<PA3> pa(m, "ParArray3D", pybind11::buffer_protocol());
  pa.def_buffer([](PA3 &arr) -> pybind11::buffer_info {
    const int size1 = arr.GetDim(1);
    const int size2 = arr.GetDim(2);
    const int size3 = arr.GetDim(3);
    return pybind11::buffer_info(
        arr.data(), sizeof(Real), pybind11::format_descriptor<Real>::format(), 3,
        {size1, size2, size3},
        {sizeof(Real), sizeof(Real) * size1, sizeof(Real) * size1 * size2});
  });

  using arg = pybind11::arg;
  pybind11::classh<SparsePack_py> pack(m, "SparsePack");
  pack.def(pybind11::init<MeshBlock *, const std::vector<std::string>>());
  pack.def("GetParArray3D", &SparsePack_py::GetParArray3D,
           "Get pointer to the underlying ParArray.", arg("block"), arg("var"), arg("te"),
           arg("comp") = 0);
  pack.def("GetCoordinates", &SparsePack_py::GetCoordinates);
}

void meshblock(pybind11::module_ &m) {
  pybind11::classh<kamayan::MeshBlock> mb(m, "MeshBlock");

  mb.def("pack",
         [](MeshBlock &self, const std::vector<std::string> &vars) -> SparsePack_py {
           return SparsePack_py(&self, vars);
         });
  mb.def("get_package", [](MeshBlock &self, const std::string &name) {
    return self.packages.Get(name);
  });
}

void grid_module(pybind11::module_ &m) {
  meshblock(m);
  sparse_pack_py(m);

  pybind11::native_enum<TopologicalElement> te(m, "TopologicalElement", "enum.Enum");
  te.value("CC", TopologicalElement::CC);
  te.value("F1", TopologicalElement::F1);
  te.value("F2", TopologicalElement::F2);
  te.value("F3", TopologicalElement::F3);
  te.value("E1", TopologicalElement::E1);
  te.value("E2", TopologicalElement::E2);
  te.value("E3", TopologicalElement::E3);
  te.value("NN", TopologicalElement::NN);
  te.finalize();

  using Coordinates_t = parthenon::Coordinates_t;
  pybind11::classh<Coordinates_t> coords(m, "Coordinates_t");
  coords.def("Dx", [](Coordinates_t &self, const int dir) { return self.Dx(dir); });
  coords.def("Dx1", &Coordinates_t::Dx<1>);
  coords.def("Dx2", &Coordinates_t::Dx<2>);
  coords.def("Dx3", &Coordinates_t::Dx<3>);

  coords.def("Xc",
             pybind11::vectorize([](Coordinates_t &self, const int &dir, const int &idx) {
               return (dir == 1) * self.Xc<1>(idx) + (dir == 2) * self.Xc<2>(idx) +
                      (dir == 3) * self.Xc<3>(idx);
             }));
  coords.def("Xc1", pybind11::vectorize([](Coordinates_t &self, const int &idx) {
               return self.Xc<1>(idx);
             }));
  coords.def("Xc2", pybind11::vectorize([](Coordinates_t &self, const int &idx) {
               return self.Xc<2>(idx);
             }));
  coords.def("Xc3", pybind11::vectorize([](Coordinates_t &self, const int &idx) {
               return self.Xc<3>(idx);
             }));
}
}  // namespace kamayan
