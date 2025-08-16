#include "grid/pybind/grid_bindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <pack/make_pack_descriptor.hpp>
#include <pack/pack_descriptor.hpp>
#include <pack/sparse_pack.hpp>

#include "coordinates/coordinates.hpp"
#include "grid/grid_types.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/sparse_pack_base.hpp"
#include "utils/parallel.hpp"
#include "utils/type_list.hpp"

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

void sparse_pack_py(nanobind::module_ &m) {
  using PA3 = parthenon::ParArray3D<Real>;
  nanobind::class_<PA3> pa(m, "ParArray3D");
  pa.def(
      "view",
      [](PA3 &self) {
        const std::size_t size1 = self.GetDim(1);
        const std::size_t size2 = self.GetDim(2);
        const std::size_t size3 = self.GetDim(3);
        return nanobind::ndarray<nanobind::numpy, Real, nanobind::ndim<3>>(
            self.data(), {size3, size2, size1});
      },
      nanobind::rv_policy::reference_internal);

  using arg = nanobind::arg;
  nanobind::class_<SparsePack_py> pack(m, "SparsePack");
  pack.def("__init__",
           [](SparsePack_py *self, MeshBlock *mb, const std::vector<std::string> &vars) {
             new (self) SparsePack_py(mb, vars);
           });
  pack.def("GetParArray3D", &SparsePack_py::GetParArray3D,
           "Get pointer to the underlying ParArray.", arg("block"), arg("var"), arg("te"),
           arg("comp") = 0);
  pack.def("GetCoordinates", &SparsePack_py::GetCoordinates);
}

void meshblock(nanobind::module_ &m) {
  nanobind::class_<kamayan::MeshBlock> mb(m, "MeshBlock");

  mb.def("pack",
         [](MeshBlock &self, const std::vector<std::string> &vars) -> SparsePack_py {
           return SparsePack_py(&self, vars);
         });
  mb.def("get_package", [](MeshBlock &self, const std::string &name) {
    return self.packages.Get(name);
  });
}

template <typename T, typename ClassMethod, typename Out_t, typename Vector_t,
          typename... Args>
void Vectorize(nanobind::class_<T> &py_class, const std::string &name, ClassMethod &&cm,
               TypeList<Out_t, Vector_t, Args...>) {
  // not sure what happens when these should exist on device...
  // we would want to be sure to return a device allocated array
  // I guess you want to construct a view with the flattened size of
  // the input array and return it as an nanobind::ndarray
  // maybe it should be a ndarray<Out_t, nanobind::cupy> {
  // and launch a par_for to fill it...
  //
  // bind the scalar
  py_class.def(name.c_str(),
               [=](T &self, Vector_t v, Args... args) { return cm(self, v, args...); });
  // bind the vector
  py_class.def(name.c_str(), [=](T &self, nanobind::ndarray<Vector_t> v, Args... args) {
    auto result = std::vector<Out_t>(v.size());
    Out_t *result_ptr = result.data();
    const Vector_t *v_ptr = v.data();
    par_for(
        PARTHENON_AUTO_LABEL, 0, v.size() - 1,
        KOKKOS_LAMBDA(const int &i) { result_ptr[i] = cm(self, v_ptr[i], args...); });
    std::vector<size_t> new_shape(v.shape_ptr(), v.shape_ptr() + v.ndim());
    return nanobind::ndarray<nanobind::numpy, Out_t>(result.data(), v.ndim(),
                                                     new_shape.data())
        .cast();
  });
}

void grid_module(nanobind::module_ &m) {
  meshblock(m);
  sparse_pack_py(m);

  nanobind::enum_<TopologicalElement> te(m, "TopologicalElement", "enum.Enum");
  te.value("CC", TopologicalElement::CC);
  te.value("F1", TopologicalElement::F1);
  te.value("F2", TopologicalElement::F2);
  te.value("F3", TopologicalElement::F3);
  te.value("E1", TopologicalElement::E1);
  te.value("E2", TopologicalElement::E2);
  te.value("E3", TopologicalElement::E3);
  te.value("NN", TopologicalElement::NN);

  using Coordinates_t = parthenon::Coordinates_t;
  nanobind::class_<Coordinates_t> coords(m, "Coordinates_t");
  coords.def("Dx", [](Coordinates_t &self, const int dir) { return self.Dx(dir); });
  coords.def("Dx1", &Coordinates_t::Dx<1>);
  coords.def("Dx2", &Coordinates_t::Dx<2>);
  coords.def("Dx3", &Coordinates_t::Dx<3>);

  Vectorize(
      coords, "Xc",
      KOKKOS_LAMBDA(const Coordinates_t &self, const int &idx, const int &dir) {
        return (dir == 1) * self.Xc<1>(idx) + (dir == 2) * self.Xc<2>(idx) +
               (dir == 3) * self.Xc<3>(idx);
      },
      TypeList<Real, int, int>());
  Vectorize(
      coords, "Xc1",
      KOKKOS_LAMBDA(const Coordinates_t &self, const int &idx) {
        return self.Xc<1>(idx);
      },
      TypeList<Real, int>());
  Vectorize(
      coords, "Xc2",
      KOKKOS_LAMBDA(const Coordinates_t &self, const int &idx) {
        return self.Xc<2>(idx);
      },
      TypeList<Real, int>());
  Vectorize(
      coords, "Xc3",
      KOKKOS_LAMBDA(const Coordinates_t &self, const int &idx) {
        return self.Xc<3>(idx);
      },
      TypeList<Real, int>());
}
}  // namespace kamayan
