#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/indexer.hpp"
#include "kamayan/config.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/reconstruction.hpp"
#include "physics/hydro/riemann_solver.hpp"
#include "physics/physics_types.hpp"
#include "tasks/tasks.hpp"
#include "utils/instrument.hpp"
#include "utils/parallel.hpp"
#include "utils/type_abstractions.hpp"

namespace kamayan::hydro {

struct CalculateFluxes {
  using options = OptTypeList<HydroFactory, ReconstructionFactory, RiemannOptions>;
  using value = TaskStatus;

  using TE = TopologicalElement;

  template <typename hydro_traits, typename reconstruction_traits, RiemannSolver riemann>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    using conserved_vars = typename hydro_traits::Conserved;
    using reconstruct_vars = typename hydro_traits::Reconstruct;
    // --8<-- [start:pack]
    auto pack_recon = grid::GetPack(reconstruct_vars(), md);
    auto pack_flux = grid::GetPack(conserved_vars(), md, {PDOpt::WithFluxes});
    // --8<-- [end:pack]

    const int ndim = md->GetNDim();
    const int nblocks = pack_recon.GetNBlocks();
    auto ib = md->GetBoundsI(IndexDomain::interior);
    auto jb = md->GetBoundsJ(IndexDomain::interior);
    auto kb = md->GetBoundsK(IndexDomain::interior);
    if constexpr (hydro_traits::MHD == Mhd::ct) {
      // need fluxes along additional dimension for edge emfs
      const int k1d = ndim > 1 ? 1 : 0;
      const int k2d = ndim > 1 ? 1 : 0;
      const int k3d = ndim > 2 ? 1 : 0;
      ib.s -= k1d;
      ib.e += k1d;
      jb.s -= k2d;
      jb.e += k2d;
      kb.s -= k3d;
      kb.e += k3d;
    }

    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int nxb = pmb->cellbounds.ncellsi(IndexDomain::entire);

    const int scratch_level = 1;  // 0 small
    const int nrecon = pack_recon.GetMaxNumberOfVars();
    size_t pencil_scratch_size_in_bytes = ScratchPad2D::shmem_size(nrecon, nxb);

    parthenon::par_for_outer(
        PARTHENON_AUTO_LABEL, 2 * pencil_scratch_size_in_bytes, scratch_level, 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k,
                      const int j) {
          // holds reconstructed vars at i - 1/2
          ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
          // holds reconstructed vars at i + 1/2
          ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);

          // --8<-- [start:rea]
          parthenon::par_for_inner(
              member, 0, nrecon - 1, ib.s - 1, ib.e + 1, [&](const int var, const int i) {
                // --8<-- [start:make-stncl]
                auto stencil =
                    MakePackStencil1D<Axis::IAXIS>(pack_recon, b, var, k, j, i);
                Reconstruct<reconstruction_traits>(stencil, vM(var, i), vP(var, i));
                // --8<-- [end:make-stncl]
              });

          member.team_barrier();
          parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            // riemann solve
            auto vL = MakeScratchIndexer(pack_recon, vP, b, i - 1);
            auto vR = MakeScratchIndexer(pack_recon, vM, b, i);
            auto pack_indexer = MakePackIndexer(pack_flux, b, k, j, i);
            RiemannFlux<TE::F1, riemann, hydro_traits>(pack_indexer, vL, vR);
          });
          // --8<-- [end:rea]
        });

    if (ndim > 1) {
      parthenon::par_for_outer(
          PARTHENON_AUTO_LABEL, 3 * pencil_scratch_size_in_bytes, scratch_level, 0,
          nblocks - 1, kb.s, kb.e,
          KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
            // Reconstruction is over the cell [j-1/2, j+1/2] centered at j
            // So we have an extra pencil for the previous reconstruction so that
            // at the face j-1/2 we can use
            //   * vL = vP_{j-1}
            //   * vR = vM_{j}
            ScratchPad2D vMP(member.team_scratch(scratch_level), nrecon, nxb);
            ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
            ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);
            // loop over flux pencils at j - 1/2
            for (int j = jb.s - 1; j <= jb.e + 1; j++) {
              parthenon::par_for_inner(
                  member, 0, nrecon - 1, ib.s, ib.e, [&](const int var, const int i) {
                    auto stencil =
                        MakePackStencil1D<Axis::JAXIS>(pack_recon, b, var, k, j, i);
                    Reconstruct<reconstruction_traits>(stencil, vM(var, i), vP(var, i));
                  });
              member.team_barrier();
              // first iteration we don't calculate fluxes, it was just for the
              // reconstruction
              if (j > jb.s - 1) {
                parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
                  // riemann solver
                  auto vL = MakeScratchIndexer(pack_recon, vMP, b, i);
                  auto vR = MakeScratchIndexer(pack_recon, vM, b, i);
                  auto pack_indexer = MakePackIndexer(pack_flux, b, k, j, i);
                  RiemannFlux<TE::F2, riemann, hydro_traits>(pack_indexer, vL, vR);
                });
              }

              auto *tmp = vMP.data();
              vMP.assign_data(vP.data());
              vP.assign_data(tmp);
            }
          });
    }

    if (ndim > 2) {
      parthenon::par_for_outer(
          PARTHENON_AUTO_LABEL, 4 * pencil_scratch_size_in_bytes, scratch_level, 0,
          nblocks - 1, jb.s, jb.e,
          KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int j) {
            // Reconstruction is over the cell [k-1/2, k+1/2] centered at k
            // So we have an extra pencil for the previous reconstruction so that
            // at the face k-1/2 we can use
            //   * vL = vP_{k-1} = vMP (v-minus-plus, vP from the previous iteration)
            //   * vR = vM_{k} = vM
            ScratchPad2D vMP(member.team_scratch(scratch_level), nrecon, nxb);
            ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
            ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);
            // loop over flux pencils at k - 1/2
            for (int k = kb.s - 1; k <= kb.e + 1; k++) {
              parthenon::par_for_inner(
                  member, 0, nrecon - 1, ib.s, ib.e, [&](const int var, const int i) {
                    auto stencil =
                        MakePackStencil1D<Axis::KAXIS>(pack_recon, b, var, k, j, i);
                    Reconstruct<reconstruction_traits>(stencil, vM(var, i), vP(var, i));
                  });
              member.team_barrier();

              if (k > kb.s - 1) {
                parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
                  // riemann solve
                  auto vL = MakeScratchIndexer(pack_recon, vMP, b, i);
                  auto vR = MakeScratchIndexer(pack_recon, vM, b, i);
                  auto pack_indexer = MakePackIndexer(pack_flux, b, k, j, i);
                  RiemannFlux<TE::F2, riemann, hydro_traits>(pack_indexer, vL, vR);
                });
              }
              auto *tmp = vMP.data();
              vMP.assign_data(vP.data());
              vP.assign_data(tmp);
            }
          });
    }

    return TaskStatus::complete;
  }
};

template <EMFAveraging emf_averaging, typename stencil_2d>
requires(emf_averaging == EMFAveraging::arithmetic)
KOKKOS_INLINE_FUNCTION Real GetEdgeEMF(stencil_2d data) {
  using TE = TopologicalElement;
  // TODO(acreyes) : should really use axis[12] from the stencil to determine the face &
  // components of MAGC that we use in the calculation
  const Real emf =
      0.25 * (data.flux(MAGC(0), 0, -1, TE::F2) + data.flux(MAGC(0), 0, 0, TE::F2) -
              data.flux(MAGC(1), -1, 0, TE::F1) - data.flux(MAGC(1), 0, 0, TE::F1));
  return emf;
}

struct CalculateEMF {
  using options = OptTypeList<MhdOptions, EMFOptions>;
  using value = TaskStatus;

  using TE = TopologicalElement;

  template <Mhd mhd, EMFAveraging emf_averaging>
  value dispatch(MeshData *md) {
    if constexpr (mhd == Mhd::ct) {
      auto pack = grid::GetPack<MAGC, MAG>(md, {PDOpt::WithFluxes});

      const int ndim = md->GetNDim();
      if (ndim < 2) return TaskStatus::complete;

      const int nblocks = pack.GetNBlocks();
      auto ib = md->GetBoundsI(IndexDomain::interior, TE::E3);
      auto jb = md->GetBoundsJ(IndexDomain::interior, TE::E3);
      auto kb = md->GetBoundsK(IndexDomain::interior, TE::E3);

      par_for(
          PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            pack.flux(b, TE::E3, MAG(), k, j, i) = GetEdgeEMF<emf_averaging>(
                MakePackStencil2D<Axis::IAXIS, Axis::JAXIS>(pack, b, 0, k, j, i));
          });
    }
    return TaskStatus::complete;
  }
};

TaskID AddFluxTasks(TaskID prev, TaskList &tl, MeshData *md) {
  // calculate fluxes -- CalculateFluxes

  // needs to return task id from last task
  // --8<-- [start:add_task]
  auto get_fluxes = tl.AddTask(
      prev, "hydro::CalculateFluxes",
      [](MeshData *md) {
        auto cfg = GetConfig(md);
        return Dispatcher<CalculateFluxes>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
      },
      md);
  // --8<-- [end:add_task]
  auto get_emf = tl.AddTask(
      get_fluxes, "hydro::CalculateEMF",
      [](MeshData *md) {
        auto cfg = GetConfig(md);
        return Dispatcher<CalculateEMF>(PARTHENON_AUTO_LABEL, cfg.get()).execute(md);
      },
      md);

  return get_emf;
}
}  // namespace kamayan::hydro
