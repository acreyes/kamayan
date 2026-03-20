#include "grid/pybind/grid_bindings.hpp"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <pack/make_pack_descriptor.hpp>
#include <pack/pack_descriptor.hpp>
#include <pack/sparse_pack.hpp>

#include "kamayan/pybind/kamayan_bindings.hpp"
#include "kamayan/pybind/kamayan_nanobind.h"

#include "grid/geometry.hpp"
#include "grid/grid_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan_utils/parallel.hpp"
#include "kamayan_utils/type_list.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/pack_utils.hpp"
#include "pack/sparse_pack_base.hpp"

namespace kamayan {

struct SparsePack_py {
  SparsePack_py(MeshBlock *mb, const std::vector<std::string> vars) {
    auto pkg = mb->resolved_packages.get();
    auto desc = parthenon::MakePackDescriptor(pkg, vars, {}, {PDOpt::WithFluxes});
    pack = desc.GetPack(mb->meshblock_data.Get().get());
    map = desc.GetMap();
    config_ = GetConfig(mb);
  }

  SparsePack_py(MeshData *md, const std::vector<std::string> vars,
                const std::set<parthenon::PDOpt> &options) {
    auto desc = parthenon::MakePackDescriptor(
        md->GetMeshPointer()->resolved_packages.get(), vars, {}, options);
    pack = desc.GetPack(md);
    map = desc.GetMap();
    config_ = GetConfig(md);
  }

  parthenon::ParArray3D<Real> GetParArray3D(const int &block, const std::string &var,
                                            const parthenon::TopologicalElement &te,
                                            const int &comp = 0) {
    auto idx = parthenon::PackIdx(map[var]);
    return pack(block, te, idx + comp);
  }

  grid::GenericCoordinate GetCoordinates(const int b) const {
    auto cfg = config_.lock();
    const Geometry geometry = cfg->Get<Geometry>();
    return grid::GenericCoordinate(geometry, pack.GetCoordinates(b));
  }

 private:
  parthenon::SparsePack<> pack;
  parthenon::SparsePackIdxMap map;
  std::weak_ptr<Config> config_;
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

  m.def("GetConfig", [](MeshBlock *mb) { return GetConfig(mb); });
  m.def("GetConfig", [](MeshData *md) { return GetConfig(md); });

  nanobind::enum_<TopologicalElement> te(m, "TopologicalElement", "enum.Enum");
  te.value("CC", TopologicalElement::CC);
  te.value("F1", TopologicalElement::F1);
  te.value("F2", TopologicalElement::F2);
  te.value("F3", TopologicalElement::F3);
  te.value("E1", TopologicalElement::E1);
  te.value("E2", TopologicalElement::E2);
  te.value("E3", TopologicalElement::E3);
  te.value("NN", TopologicalElement::NN);

  nanobind::class_<grid::GenericCoordinate> gen_coords(m, "GenericCoordinate");

  gen_coords.def(
      "Dx", [](grid::GenericCoordinate &self, const Axis &ax) { return self.Dx(ax); });
  gen_coords.def("Dx1",
                 [](grid::GenericCoordinate &self) { return self.Dx<Axis::IAXIS>(); });
  gen_coords.def("Dx2",
                 [](grid::GenericCoordinate &self) { return self.Dx<Axis::JAXIS>(); });
  gen_coords.def("Dx3",
                 [](grid::GenericCoordinate &self) { return self.Dx<Axis::KAXIS>(); });

  Vectorize(
      gen_coords, "Xc",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx, const int &dir) {
        return (dir == 1) * self.Xc<Axis::IAXIS>(idx) +
               (dir == 2) * self.Xc<Axis::JAXIS>(idx) +
               (dir == 3) * self.Xc<Axis::KAXIS>(idx);
      },
      TypeList<Real, int, int>());
  Vectorize(
      gen_coords, "Xf",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx, const int &dir) {
        return (dir == 1) * self.Xf<Axis::IAXIS>(idx) +
               (dir == 2) * self.Xf<Axis::JAXIS>(idx) +
               (dir == 3) * self.Xf<Axis::KAXIS>(idx);
      },
      TypeList<Real, int, int>());
  Vectorize(
      gen_coords, "Xc1",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx) {
        return self.Xc<Axis::IAXIS>(idx);
      },
      TypeList<Real, int>());
  Vectorize(
      gen_coords, "Xc2",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx) {
        return self.Xc<Axis::JAXIS>(idx);
      },
      TypeList<Real, int>());
  Vectorize(
      gen_coords, "Xc3",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx) {
        return self.Xc<Axis::KAXIS>(idx);
      },
      TypeList<Real, int>());
  Vectorize(
      gen_coords, "Xf1",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx) {
        return self.Xf<Axis::IAXIS>(idx);
      },
      TypeList<Real, int>());
  Vectorize(
      gen_coords, "Xf2",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx) {
        return self.Xf<Axis::JAXIS>(idx);
      },
      TypeList<Real, int>());
  Vectorize(
      gen_coords, "Xf3",
      KOKKOS_LAMBDA(const grid::GenericCoordinate &self, const int &idx) {
        return self.Xf<Axis::KAXIS>(idx);
      },
      TypeList<Real, int>());

  gen_coords.def("FaceArea", [](grid::GenericCoordinate &self, const Axis &ax, int k,
                                int j, int i) { return self.FaceArea(ax, k, j, i); });
  gen_coords.def("EdgeLength", [](grid::GenericCoordinate &self, const Axis &ax, int k,
                                  int j, int i) { return self.EdgeLength(ax, k, j, i); });
  gen_coords.def("CellVolume", &grid::GenericCoordinate::CellVolume);
  gen_coords.def("Volume", &grid::GenericCoordinate::Volume);
}
}  // namespace kamayan
