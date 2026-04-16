#include <iostream>
#include <type_traits>

#include "basic_types.hpp"
#include "dispatcher/options.hpp"
#include "driver/kamayan_driver_types.hpp"
#include "grid/grid.hpp"
#include "grid/grid_types.hpp"
#include "grid/indexer.hpp"
#include "grid/subpack.hpp"
#include "hydro_types.hpp"
#include "kamayan/config.hpp"
#include "kamayan/fields.hpp"
#include "kamayan_utils/parallel.hpp"
#include "kamayan_utils/type_abstractions.hpp"
#include "kamayan_utils/type_list.hpp"
#include "kokkos_abstraction.hpp"
#include "physics/hydro/hydro.hpp"
#include "physics/hydro/hydro_types.hpp"
#include "physics/hydro/reconstruction.hpp"
#include "physics/hydro/riemann_solver.hpp"
#include "physics/hydro/thinc.hpp"
#include "physics/physics_types.hpp"
#include "utils/instrument.hpp"

namespace kamayan::hydro {

struct CalculateFluxesNested {
  using options = OptTypeList<HydroFactory, ReconstructionFactory, RiemannOptions>;
  using value = TaskStatus;

  using TE = TopologicalElement;

  template <HydroTrait hydro_traits, ReconstructTrait reconstruction_traits,
            RiemannSolver riemann>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    // could also pack the mass scalars separately...
    using conserved_vars = ConcatTypeLists_t<typename hydro_traits::Conserved,
                                             typename hydro_traits::MassScalars>;
    // include mass scalars, since the riemann states can be allocated
    // dynamically in the scratchpad
    using reconstruct_vars = ConcatTypeLists_t<typename hydro_traits::Reconstruct,
                                               typename hydro_traits::MassScalars>;
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

    if constexpr (reconstruction_traits::slope_limiter == SlopeLimiter::thinc) {
      // ---- THINC+BVD path ----
      auto hydro_pkg = md->GetMeshPointer()->packages.Get("hydro");
      const Real beta_thinc = hydro_pkg->template Param<Real>("hydro/beta_thinc");
      const bool thinc_dens = hydro_pkg->template Param<bool>("hydro/thinc_dens");
      const bool thinc_eint = hydro_pkg->template Param<bool>("hydro/thinc_eint");
      const Real thinc_threshold = hydro_pkg->template Param<Real>("hydro/thinc_threshold");
      auto cfg = GetConfig(md);
      auto fallback = cfg->Get<ThincFallbackLimiter>();

      auto pack_sensor = grid::GetPack<THINC_SENSOR>(md);

      // Zero the THINC sensor field before flux calculation
      parthenon::par_for(
          PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            pack_sensor(b, THINC_SENSOR(), k, j, i) = 0.0;
          });

      auto run_thinc = [&]<ThincFallbackLimiter fb_limiter>() {
        constexpr SlopeLimiter fb_sl = to_slope_limiter<fb_limiter>();
        using FBTraits =
            ReconstructTraits<reconstruction_traits::reconstruction, fb_sl>;

        // ---- I-axis (6 scratch pads) ----
        {
          constexpr int n_pencils = 6;
          parthenon::par_for_outer(
              PARTHENON_AUTO_LABEL, n_pencils * pencil_scratch_size_in_bytes,
              scratch_level, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
              KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b,
                            const int k, const int j) {
                ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
                ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);
                ScratchPad2D thinc_vM(member.team_scratch(scratch_level), nrecon,
                                      nxb);
                ScratchPad2D thinc_vP(member.team_scratch(scratch_level), nrecon,
                                      nxb);
                ScratchPad2D sel_vM(member.team_scratch(scratch_level), nrecon,
                                    nxb);
                ScratchPad2D sel_vP(member.team_scratch(scratch_level), nrecon,
                                    nxb);

                const int dens_idx = pack_recon.GetLowerBound(b, DENS());
                const int eint_idx = pack_recon.GetLowerBound(b, EINT());

                // Pass 1: fallback + THINC reconstruction
                parthenon::par_for_inner(
                    member, 0, nrecon - 1, ib.s - 1, ib.e + 1,
                    [&](const int var, const int i) {
                      const bool is_tv = (thinc_dens && var == dens_idx) ||
                                         (thinc_eint && var == eint_idx);
                      ReconstructFace<FBTraits, Axis::IAXIS>(
                          pack_recon, b, var, k, j, i, vM(var, i), vP(var, i),
                          thinc_vM(var, i), thinc_vP(var, i), beta_thinc,
                          is_tv);
                    });

                member.team_barrier();

                // Pass 2: BVD selection — write to sel_vM, sel_vP
                parthenon::par_for_inner(
                    member, 0, nrecon - 1, ib.s, ib.e + 1,
                    [&](const int var, const int i) {
                      const bool is_tv = (thinc_dens && var == dens_idx) ||
                                         (thinc_eint && var == eint_idx);
                      bool use_thinc = false;
                      if (is_tv) {
                        // BVD at face i:
                        //   fb_L = vP(i-1), fb_R = vM(i)
                        //   th_L = thinc_vP(i-1), th_R = thinc_vM(i)
                        //   fb_Lm = vM(i-1) [R-state at face i-1]
                        //   th_Lm = thinc_vM(i-1)
                        //   fb_Rp = vP(i) [L-state at face i+1]
                        //   th_Rp = thinc_vP(i)
                        use_thinc = BVDSelect(
                            vP(var, i - 1), vM(var, i), thinc_vP(var, i - 1),
                            thinc_vM(var, i), vM(var, i - 1),
                            thinc_vM(var, i - 1), vP(var, i),
                            thinc_vP(var, i), thinc_threshold);
                      }
                      if (use_thinc && var == dens_idx) {
                        pack_sensor(b, THINC_SENSOR(), k, j, i) = 1.0;
                        pack_sensor(b, THINC_SENSOR(), k, j, i - 1) = 1.0;
                      }
                      sel_vP(var, i - 1) = use_thinc ? thinc_vP(var, i - 1)
                                                      : vP(var, i - 1);
                      sel_vM(var, i) =
                          use_thinc ? thinc_vM(var, i) : vM(var, i);
                    });

                member.team_barrier();

                // Riemann solve (reads from sel_vP, sel_vM)
                parthenon::par_for_inner(member, ib.s, ib.e + 1,
                                         [&](const int i) {
                                           auto vL = MakeScratchIndexer(
                                               pack_recon, sel_vP, b, i - 1);
                                           auto vR = MakeScratchIndexer(
                                               pack_recon, sel_vM, b, i);
                                           auto pack_indexer =
                                               SubPack(pack_flux, b, k, j, i);
                                           if constexpr (hydro_traits::MHD ==
                                                         Mhd::ct) {
                                             vL(MAGC(0)) =
                                                 pack_indexer(TE::F1, MAG());
                                             vR(MAGC(0)) =
                                                 pack_indexer(TE::F1, MAG());
                                           }
                                           RiemannFlux<TE::F1, riemann,
                                                       hydro_traits>(
                                               pack_indexer, vL, vR);
                                         });

                member.team_barrier();
                type_for(typename hydro_traits::MassScalars(),
                         [&]<typename V>(const V &v) {
                           int offset = count_components(
                               typename hydro_traits::Reconstruct());
                           for (int s = 0;
                                s <= pack_flux.GetUpperBound(b, V()); s++) {
                             par_for_inner(
                                 member, ib.s, ib.e + 1, [&](const int i) {
                                   const auto rho_flux = pack_flux.flux(
                                       b, TE::F1, DENS(), k, j, i);
                                   pack_flux.flux(b, TE::F1, V(s), k, j, i) =
                                       rho_flux > 0.0
                                           ? rho_flux *
                                                 sel_vP(offset + s, i - 1)
                                           : rho_flux *
                                                 sel_vM(offset + s, i);
                                 });
                           }
                           offset++;
                         });
              });
        }

        // ---- J-axis (8 scratch pads) ----
        if (ndim > 1) {
          constexpr int n_pencils_j = 8;
          parthenon::par_for_outer(
              PARTHENON_AUTO_LABEL, n_pencils_j * pencil_scratch_size_in_bytes,
              scratch_level, 0, nblocks - 1, kb.s, kb.e,
              KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b,
                            const int k) {
                ScratchPad2D vMP(member.team_scratch(scratch_level), nrecon,
                                 nxb);
                ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
                ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);
                ScratchPad2D thinc_vMP(member.team_scratch(scratch_level),
                                       nrecon, nxb);
                ScratchPad2D thinc_vM(member.team_scratch(scratch_level), nrecon,
                                      nxb);
                ScratchPad2D thinc_vP(member.team_scratch(scratch_level), nrecon,
                                      nxb);
                ScratchPad2D prev_fb_vM(member.team_scratch(scratch_level),
                                        nrecon, nxb);
                ScratchPad2D prev_thinc_vM(member.team_scratch(scratch_level),
                                           nrecon, nxb);

                const int dens_idx = pack_recon.GetLowerBound(b, DENS());
                const int eint_idx = pack_recon.GetLowerBound(b, EINT());

                for (int j = jb.s - 1; j <= jb.e + 1; j++) {
                  // Reconstruct
                  parthenon::par_for_inner(
                      member, 0, nrecon - 1, ib.s, ib.e,
                      [&](const int var, const int i) {
                        const bool is_tv = (thinc_dens && var == dens_idx) ||
                                           (thinc_eint && var == eint_idx);
                        ReconstructFace<FBTraits, Axis::JAXIS>(
                            pack_recon, b, var, k, j, i, vM(var, i),
                            vP(var, i), thinc_vM(var, i), thinc_vP(var, i),
                            beta_thinc, is_tv);
                      });
                  member.team_barrier();

                  if (j > jb.s - 1) {
                    // BVD selection + save prev_fb_vM/prev_thinc_vM
                    // Face j-1/2:
                    //   fb_L = vMP(i) [= vP from j-1]
                    //   fb_R = vM(i) [= vM from j]
                    //   th_L = thinc_vMP(i) [= thinc_vP from j-1]
                    //   th_R = thinc_vM(i) [= thinc_vM from j]
                    //   fb_Lm = prev_fb_vM(i) [= vM from j-1 = R at face j-3/2]
                    //   th_Lm = prev_thinc_vM(i)
                    //   fb_Rp = vP(i) [= L at face j+1/2]
                    //   th_Rp = thinc_vP(i)
                    parthenon::par_for_inner(
                        member, 0, nrecon - 1, ib.s, ib.e,
                        [&](const int var, const int i) {
                          const bool is_tv = (thinc_dens && var == dens_idx) ||
                                             (thinc_eint && var == eint_idx);
                          // Save raw values before BVD modifies vMP/vM
                          Real raw_vM = vM(var, i);
                          Real raw_thinc_vM = thinc_vM(var, i);

                          if (is_tv) {
                            bool use_thinc = BVDSelect(
                                vMP(var, i), vM(var, i), thinc_vMP(var, i),
                                thinc_vM(var, i), prev_fb_vM(var, i),
                                prev_thinc_vM(var, i), vP(var, i),
                                thinc_vP(var, i), thinc_threshold);
                            if (use_thinc) {
                              vMP(var, i) = thinc_vMP(var, i);
                              vM(var, i) = thinc_vM(var, i);
                            }
                            if (use_thinc && var == dens_idx) {
                              pack_sensor(b, THINC_SENSOR(), k, j, i) = 1.0;
                              pack_sensor(b, THINC_SENSOR(), k, j - 1, i) = 1.0;
                            }
                          }

                          prev_fb_vM(var, i) = raw_vM;
                          prev_thinc_vM(var, i) = raw_thinc_vM;
                        });
                    member.team_barrier();

                    // Riemann solve (reads from potentially-modified vMP, vM)
                    parthenon::par_for_inner(member, ib.s, ib.e,
                                             [&](const int i) {
                                               auto vL = MakeScratchIndexer(
                                                   pack_recon, vMP, b, i);
                                               auto vR = MakeScratchIndexer(
                                                   pack_recon, vM, b, i);
                                               auto pack_indexer = SubPack(
                                                   pack_flux, b, k, j, i);
                                               if constexpr (hydro_traits::MHD ==
                                                             Mhd::ct) {
                                                 vL(MAGC(1)) = pack_indexer(
                                                     TE::F2, MAG());
                                                 vR(MAGC(1)) = pack_indexer(
                                                     TE::F2, MAG());
                                               }
                                               RiemannFlux<TE::F2, riemann,
                                                           hydro_traits>(
                                                   pack_indexer, vL, vR);
                                             });

                    member.team_barrier();
                    type_for(
                        typename hydro_traits::MassScalars(),
                        [&]<typename V>(const V &v) {
                          int offset = count_components(
                              typename hydro_traits::Reconstruct());
                          for (int s = 0; s < pack_flux.GetUpperBound(b, V());
                               s++) {
                            par_for_inner(
                                member, ib.s, ib.e, [&](const int i) {
                                  const auto rho_flux = pack_flux.flux(
                                      b, TE::F2, DENS(), k, j, i);
                                  pack_flux.flux(b, TE::F2, V(s), k, j, i) =
                                      rho_flux > 0.0
                                          ? rho_flux * vMP(offset + s, i)
                                          : rho_flux * vM(offset + s, i);
                                });
                          }
                          offset++;
                        });
                  } else {
                    // First iteration: just save prev data for next
                    // iteration's BVD
                    parthenon::par_for_inner(
                        member, 0, nrecon - 1, ib.s, ib.e,
                        [&](const int var, const int i) {
                          prev_fb_vM(var, i) = vM(var, i);
                          prev_thinc_vM(var, i) = thinc_vM(var, i);
                        });
                    member.team_barrier();
                  }

                  // Pointer swap: vMP ↔ vP
                  auto *tmp = vMP.data();
                  vMP.assign_data(vP.data());
                  vP.assign_data(tmp);
                  // Pointer swap: thinc_vMP ↔ thinc_vP
                  auto *thinc_tmp = thinc_vMP.data();
                  thinc_vMP.assign_data(thinc_vP.data());
                  thinc_vP.assign_data(thinc_tmp);
                }
              });
        }

        // ---- K-axis (8 scratch pads) ----
        if (ndim > 2) {
          constexpr int n_pencils_k = 8;
          parthenon::par_for_outer(
              PARTHENON_AUTO_LABEL, n_pencils_k * pencil_scratch_size_in_bytes,
              scratch_level, 0, nblocks - 1, jb.s, jb.e,
              KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b,
                            const int j) {
                ScratchPad2D vMP(member.team_scratch(scratch_level), nrecon,
                                 nxb);
                ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
                ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);
                ScratchPad2D thinc_vMP(member.team_scratch(scratch_level),
                                       nrecon, nxb);
                ScratchPad2D thinc_vM(member.team_scratch(scratch_level), nrecon,
                                      nxb);
                ScratchPad2D thinc_vP(member.team_scratch(scratch_level), nrecon,
                                      nxb);
                ScratchPad2D prev_fb_vM(member.team_scratch(scratch_level),
                                        nrecon, nxb);
                ScratchPad2D prev_thinc_vM(member.team_scratch(scratch_level),
                                           nrecon, nxb);

                const int dens_idx = pack_recon.GetLowerBound(b, DENS());
                const int eint_idx = pack_recon.GetLowerBound(b, EINT());

                for (int k = kb.s - 1; k <= kb.e + 1; k++) {
                  // Reconstruct
                  parthenon::par_for_inner(
                      member, 0, nrecon - 1, ib.s, ib.e,
                      [&](const int var, const int i) {
                        const bool is_tv = (thinc_dens && var == dens_idx) ||
                                           (thinc_eint && var == eint_idx);
                        ReconstructFace<FBTraits, Axis::KAXIS>(
                            pack_recon, b, var, k, j, i, vM(var, i),
                            vP(var, i), thinc_vM(var, i), thinc_vP(var, i),
                            beta_thinc, is_tv);
                      });
                  member.team_barrier();

                  if (k > kb.s - 1) {
                    // BVD selection + save prev
                    parthenon::par_for_inner(
                        member, 0, nrecon - 1, ib.s, ib.e,
                        [&](const int var, const int i) {
                          const bool is_tv = (thinc_dens && var == dens_idx) ||
                                             (thinc_eint && var == eint_idx);
                          Real raw_vM = vM(var, i);
                          Real raw_thinc_vM = thinc_vM(var, i);

                          if (is_tv) {
                            bool use_thinc = BVDSelect(
                                vMP(var, i), vM(var, i), thinc_vMP(var, i),
                                thinc_vM(var, i), prev_fb_vM(var, i),
                                prev_thinc_vM(var, i), vP(var, i),
                                thinc_vP(var, i), thinc_threshold);
                            if (use_thinc) {
                              vMP(var, i) = thinc_vMP(var, i);
                              vM(var, i) = thinc_vM(var, i);
                            }
                            if (use_thinc && var == dens_idx) {
                              pack_sensor(b, THINC_SENSOR(), k, j, i) = 1.0;
                              pack_sensor(b, THINC_SENSOR(), k - 1, j, i) = 1.0;
                            }
                          }

                          prev_fb_vM(var, i) = raw_vM;
                          prev_thinc_vM(var, i) = raw_thinc_vM;
                        });
                    member.team_barrier();

                    // Riemann solve
                    parthenon::par_for_inner(member, ib.s, ib.e,
                                             [&](const int i) {
                                               auto vL = MakeScratchIndexer(
                                                   pack_recon, vMP, b, i);
                                               auto vR = MakeScratchIndexer(
                                                   pack_recon, vM, b, i);
                                               auto pack_indexer = SubPack(
                                                   pack_flux, b, k, j, i);
                                               if constexpr (hydro_traits::MHD ==
                                                             Mhd::ct) {
                                                 vL(MAGC(2)) = pack_indexer(
                                                     TE::F3, MAG());
                                                 vR(MAGC(2)) = pack_indexer(
                                                     TE::F3, MAG());
                                               }
                                               RiemannFlux<TE::F3, riemann,
                                                           hydro_traits>(
                                                   pack_indexer, vL, vR);
                                             });
                    member.team_barrier();

                    type_for(
                        typename hydro_traits::MassScalars(),
                        [&]<typename V>(const V &v) {
                          int offset = count_components(
                              typename hydro_traits::Reconstruct());
                          for (int s = 0; s < pack_flux.GetUpperBound(b, V());
                               s++) {
                            par_for_inner(
                                member, ib.s, ib.e + 1, [&](const int i) {
                                  const auto rho_flux = pack_flux.flux(
                                      b, TE::F3, DENS(), k, j, i);
                                  pack_flux.flux(b, TE::F3, V(s), k, j, i) =
                                      rho_flux > 0.0
                                          ? rho_flux * vMP(offset + s, i)
                                          : rho_flux * vM(offset + s, i);
                                });
                          }
                          offset++;
                        });
                  } else {
                    // First iteration: save prev data
                    parthenon::par_for_inner(
                        member, 0, nrecon - 1, ib.s, ib.e,
                        [&](const int var, const int i) {
                          prev_fb_vM(var, i) = vM(var, i);
                          prev_thinc_vM(var, i) = thinc_vM(var, i);
                        });
                    member.team_barrier();
                  }

                  // Pointer swap: vMP ↔ vP
                  auto *tmp = vMP.data();
                  vMP.assign_data(vP.data());
                  vP.assign_data(tmp);
                  // Pointer swap: thinc_vMP ↔ thinc_vP
                  auto *thinc_tmp = thinc_vMP.data();
                  thinc_vMP.assign_data(thinc_vP.data());
                  thinc_vP.assign_data(thinc_tmp);
                }
              });
        }
      };  // end run_thinc

      // Local lambda dispatch for ThincFallbackLimiter
      if (fallback == ThincFallbackLimiter::minmod)
        run_thinc.template operator()<ThincFallbackLimiter::minmod>();
      else if (fallback == ThincFallbackLimiter::van_leer)
        run_thinc.template operator()<ThincFallbackLimiter::van_leer>();
      else
        run_thinc.template operator()<ThincFallbackLimiter::mc>();

    } else {
      // ---- Standard (non-THINC) path ----

      parthenon::par_for_outer(
          PARTHENON_AUTO_LABEL, 2 * pencil_scratch_size_in_bytes, scratch_level,
          0, nblocks - 1, kb.s, kb.e, jb.s, jb.e,
          KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k,
                        const int j) {
            // holds reconstructed vars at i - 1/2
            ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
            // holds reconstructed vars at i + 1/2
            ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);

            // --8<-- [start:rea]
            parthenon::par_for_inner(
                member, 0, nrecon - 1, ib.s - 1, ib.e + 1,
                [&](const int var, const int i) {
                  // --8<-- [start:make-stncl]
                  auto stencil =
                      SubPack<Axis::IAXIS>(pack_recon, b, var, k, j, i);
                  Reconstruct<reconstruction_traits>(stencil, vM(var, i),
                                                     vP(var, i));
                  // --8<-- [end:make-stncl]
                });

            member.team_barrier();
            parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
              // riemann solve
              auto vL = MakeScratchIndexer(pack_recon, vP, b, i - 1);
              auto vR = MakeScratchIndexer(pack_recon, vM, b, i);
              auto pack_indexer = SubPack(pack_flux, b, k, j, i);
              if constexpr (hydro_traits::MHD == Mhd::ct) {
                vL(MAGC(0)) = pack_indexer(TE::F1, MAG());
                vR(MAGC(0)) = pack_indexer(TE::F1, MAG());
              }
              RiemannFlux<TE::F1, riemann, hydro_traits>(pack_indexer, vL, vR);
            });
            // --8<-- [end:rea]

            member.team_barrier();
            type_for(typename hydro_traits::MassScalars(),
                     [&]<typename V>(const V &v) {
                       int offset =
                           count_components(typename hydro_traits::Reconstruct());
                       for (int s = 0; s <= pack_flux.GetUpperBound(b, V());
                            s++) {
                         par_for_inner(member, ib.s, ib.e + 1,
                                       [&](const int i) {
                                         const auto rho_flux = pack_flux.flux(
                                             b, TE::F1, DENS(), k, j, i);

                                         pack_flux.flux(b, TE::F1, V(s), k, j,
                                                        i) =
                                             rho_flux > 0.0
                                                 ? rho_flux *
                                                       vP(offset + s, i - 1)
                                                 : rho_flux *
                                                       vM(offset + s, i);
                                       });
                       }
                       offset++;
                     });
          });

      if (ndim > 1) {
        parthenon::par_for_outer(
            PARTHENON_AUTO_LABEL, 3 * pencil_scratch_size_in_bytes,
            scratch_level, 0, nblocks - 1, kb.s, kb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b,
                          const int k) {
              ScratchPad2D vMP(member.team_scratch(scratch_level), nrecon, nxb);
              ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
              ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);
              for (int j = jb.s - 1; j <= jb.e + 1; j++) {
                parthenon::par_for_inner(
                    member, 0, nrecon - 1, ib.s, ib.e,
                    [&](const int var, const int i) {
                      auto stencil =
                          SubPack<Axis::JAXIS>(pack_recon, b, var, k, j, i);
                      Reconstruct<reconstruction_traits>(stencil, vM(var, i),
                                                         vP(var, i));
                    });
                member.team_barrier();
                if (j > jb.s - 1) {
                  parthenon::par_for_inner(member, ib.s, ib.e,
                                           [&](const int i) {
                                             auto vL = MakeScratchIndexer(
                                                 pack_recon, vMP, b, i);
                                             auto vR = MakeScratchIndexer(
                                                 pack_recon, vM, b, i);
                                             auto pack_indexer = SubPack(
                                                 pack_flux, b, k, j, i);
                                             if constexpr (hydro_traits::MHD ==
                                                           Mhd::ct) {
                                               vL(MAGC(1)) = pack_indexer(
                                                   TE::F2, MAG());
                                               vR(MAGC(1)) = pack_indexer(
                                                   TE::F2, MAG());
                                             }
                                             RiemannFlux<TE::F2, riemann,
                                                         hydro_traits>(
                                                 pack_indexer, vL, vR);
                                           });

                  member.team_barrier();
                  type_for(typename hydro_traits::MassScalars(),
                           [&]<typename V>(const V &v) {
                             int offset = count_components(
                                 typename hydro_traits::Reconstruct());
                             for (int s = 0;
                                  s < pack_flux.GetUpperBound(b, V()); s++) {
                               par_for_inner(
                                   member, ib.s, ib.e, [&](const int i) {
                                     const auto rho_flux = pack_flux.flux(
                                         b, TE::F2, DENS(), k, j, i);

                                     pack_flux.flux(b, TE::F2, V(s), k, j, i) =
                                         rho_flux > 0.0
                                             ? rho_flux * vMP(offset + s, i)
                                             : rho_flux * vM(offset + s, i);
                                   });
                             }
                             offset++;
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
            PARTHENON_AUTO_LABEL, 4 * pencil_scratch_size_in_bytes,
            scratch_level, 0, nblocks - 1, jb.s, jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b,
                          const int j) {
              ScratchPad2D vMP(member.team_scratch(scratch_level), nrecon, nxb);
              ScratchPad2D vM(member.team_scratch(scratch_level), nrecon, nxb);
              ScratchPad2D vP(member.team_scratch(scratch_level), nrecon, nxb);
              for (int k = kb.s - 1; k <= kb.e + 1; k++) {
                parthenon::par_for_inner(
                    member, 0, nrecon - 1, ib.s, ib.e,
                    [&](const int var, const int i) {
                      auto stencil =
                          SubPack<Axis::KAXIS>(pack_recon, b, var, k, j, i);
                      Reconstruct<reconstruction_traits>(stencil, vM(var, i),
                                                         vP(var, i));
                    });
                member.team_barrier();

                if (k > kb.s - 1) {
                  parthenon::par_for_inner(member, ib.s, ib.e,
                                           [&](const int i) {
                                             auto vL = MakeScratchIndexer(
                                                 pack_recon, vMP, b, i);
                                             auto vR = MakeScratchIndexer(
                                                 pack_recon, vM, b, i);
                                             auto pack_indexer = SubPack(
                                                 pack_flux, b, k, j, i);
                                             if constexpr (hydro_traits::MHD ==
                                                           Mhd::ct) {
                                               vL(MAGC(2)) = pack_indexer(
                                                   TE::F3, MAG());
                                               vR(MAGC(2)) = pack_indexer(
                                                   TE::F3, MAG());
                                             }
                                             RiemannFlux<TE::F3, riemann,
                                                         hydro_traits>(
                                                 pack_indexer, vL, vR);
                                           });
                  member.team_barrier();

                  type_for(typename hydro_traits::MassScalars(),
                           [&]<typename V>(const V &v) {
                             int offset = count_components(
                                 typename hydro_traits::Reconstruct());
                             for (int s = 0;
                                  s < pack_flux.GetUpperBound(b, V()); s++) {
                               par_for_inner(
                                   member, ib.s, ib.e + 1, [&](const int i) {
                                     const auto rho_flux = pack_flux.flux(
                                         b, TE::F3, DENS(), k, j, i);

                                     pack_flux.flux(b, TE::F3, V(s), k, j, i) =
                                         rho_flux > 0.0
                                             ? rho_flux * vMP(offset + s, i)
                                             : rho_flux * vM(offset + s, i);
                                   });
                             }
                             offset++;
                           });
                }
                auto *tmp = vMP.data();
                vMP.assign_data(vP.data());
                vP.assign_data(tmp);
              }
            });
      }
    }  // end non-THINC path

    return TaskStatus::complete;
  }
};
struct CalculateFluxesScratch {
  using options = OptTypeList<HydroFactory, ReconstructionFactory, RiemannOptions>;
  using value = TaskStatus;

  using TE = TopologicalElement;

  template <HydroTrait hydro_traits, ReconstructTrait reconstruction_traits,
            RiemannSolver riemann>
  requires(NonTypeTemplateSpecialization<hydro_traits, HydroTraits>)
  value dispatch(MeshData *md) {
    if constexpr (reconstruction_traits::slope_limiter == SlopeLimiter::thinc) {
      PARTHENON_FAIL(
          "THINC reconstruction is not yet supported with the scratchvar "
          "strategy. Use ReconstructionStrategy = scratchpad instead.");
    } else {
      using conserved_vars = ConcatTypeLists_t<typename hydro_traits::Conserved,
                                               typename hydro_traits::MassScalars>;
      using reconstruct_vars = ConcatTypeLists_t<typename hydro_traits::Reconstruct,
                                                  typename hydro_traits::MassScalars>;
      using minus = RiemannScratch::Minus;
      using plus = RiemannScratch::Plus;

      auto pack_recon = grid::GetPack(reconstruct_vars(), md);
      auto pack_flux = grid::GetPack(conserved_vars(), md, {PDOpt::WithFluxes});

      auto hydro = md->GetMeshPointer()->packages.Get("hydro");
      const auto riemann_scratch =
          hydro->Param<RiemannScratch::type>("riemann_scratch");
      auto pack_scratch = ScratchPack(md, riemann_scratch);

      const int ndim = md->GetNDim();
      const int nblocks = pack_recon.GetNBlocks();
      auto ib = md->GetBoundsI(IndexDomain::interior);
      auto jb = md->GetBoundsJ(IndexDomain::interior);
      auto kb = md->GetBoundsK(IndexDomain::interior);
      if constexpr (hydro_traits::MHD == Mhd::ct) {
        const int k2d = ndim > 1 ? 1 : 0;
        const int k3d = ndim > 2 ? 1 : 0;
        ib.s -= k2d;
        ib.e += k2d;
        jb.s -= k2d;
        jb.e += k2d;
        kb.s -= k3d;
        kb.e += k3d;
      }

      auto calc_fluxes = [&]<Axis axis>() {
        constexpr auto dir =
            axis == Axis::IAXIS ? 0 : axis == Axis::JAXIS ? 1 : 2;
        constexpr auto face = static_cast<TE>(static_cast<int>(TE::F1) + dir);

        constexpr auto ii = axis == Axis::IAXIS ? 1 : 0;
        constexpr auto jj = axis == Axis::JAXIS ? 1 : 0;
        constexpr auto kk = axis == Axis::KAXIS ? 1 : 0;

        // bounds padded along axis
        parthenon::IndexRange pad_ib{ib.s - ii, ib.e + ii};
        parthenon::IndexRange pad_jb{jb.s - jj, jb.e + jj};
        parthenon::IndexRange pad_kb{kb.s - kk, kb.e + kk};

        par_for_outer(
            PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1, pad_kb.s, pad_kb.e,
            pad_jb.s, pad_jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b,
                          const int k, const int j) {
              par_for_inner(
                  member, 0, pack_recon.GetUpperBound(b), pad_ib.s, pad_ib.e,
                  [&](const int &var, const int &i) {
                    auto stencil = SubPack<axis>(pack_recon, b, var, k, j, i);
                    auto vMP = pack_scratch.SubPack(b, k, j, i);
                    Reconstruct<reconstruction_traits>(stencil, vMP(minus(var)),
                                                       vMP(plus(var)));
                  });
            });

        par_for(
            PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, pad_kb.e, jb.s,
            pad_jb.e, ib.s, pad_ib.e,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
              auto vL = pack_scratch.Indexer(plus(), reconstruct_vars(), b,
                                             k - kk, j - jj, i - ii);
              auto vR = pack_scratch.Indexer(minus(), reconstruct_vars(), b, k,
                                             j, i);

              auto pack_indexer = SubPack(pack_flux, b, k, j, i);
              if constexpr (hydro_traits::MHD == Mhd::ct) {
                vL(MAGC(dir)) = pack_indexer(face, MAG());
                vR(MAGC(dir)) = pack_indexer(face, MAG());
              }
              RiemannFlux<face, riemann, hydro_traits>(pack_indexer, vL, vR);
            });

        par_for_outer(
            PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1, kb.s, pad_kb.e, jb.s,
            pad_jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b,
                          const int k, const int j) {
              const int offset =
                  count_components(typename hydro_traits::Reconstruct());
              par_for_inner(
                  member, offset, pack_recon.GetUpperBound(b), ib.s, pad_ib.e,
                  [&](const int &var, const int &i) {
                    const auto rho_flux =
                        pack_flux.flux(b, face, DENS(), k, j, i);
                    const int ms_idx =
                        var - offset +
                        count_components(typename hydro_traits::Conserved());
                    pack_flux.flux(b, face, ms_idx, k, j, i) =
                        rho_flux > 0.0 ? rho_flux * pack_scratch(b, plus(var),
                                                                  k - kk,
                                                                  j - jj, i - ii)
                                       : rho_flux * pack_scratch(b, minus(var),
                                                                  k, j, i);
                  });
            });
      };  // NOLINT(readability/braces)

      calc_fluxes.template operator()<Axis::IAXIS>();
      if (ndim > 1) calc_fluxes.template operator()<Axis::JAXIS>();
      if (ndim > 2) calc_fluxes.template operator()<Axis::KAXIS>();
    }  // end non-THINC path

    return TaskStatus::complete;
  }
};

template <TopologicalElement edge, EMFAveraging emf_averaging, typename stencil_2d>
requires(EdgeElement<edge> && emf_averaging == EMFAveraging::arithmetic)
KOKKOS_INLINE_FUNCTION Real GetEdgeEMF(stencil_2d data) {
  using TE = TopologicalElement;
  constexpr auto face1 = IncrementTE(TE::F1, edge, 1);
  constexpr auto b1 = static_cast<int>(face1) % 3;
  constexpr auto face2 = IncrementTE(TE::F1, edge, 2);
  constexpr auto b2 = static_cast<int>(face2) % 3;
  // Ez = -Fx(By) = Fy(Bx)
  const Real emf =
      0.25 * (data.flux(face2, MAGC(b1), -1, 0) + data.flux(face2, MAGC(b1), 0, 0) -
              data.flux(face1, MAGC(b2), 0, -1) - data.flux(face1, MAGC(b2), 0, 0));
  return emf;
}

struct CalculateEMF {
  using options = OptTypeList<MhdOptions, EMFOptions>;
  using value = TaskStatus;

  using TE = TopologicalElement;

  template <Mhd mhd, EMFAveraging emf_averaging>
  value dispatch(MeshData *md) {
    if constexpr (mhd == Mhd::ct) {
      auto pack = grid::GetPack<MAGC, MAG, EELE, EION, ERAD>(md, {PDOpt::WithFluxes});

      const int ndim = md->GetNDim();
      if (ndim < 2) return TaskStatus::complete;

      const int nblocks = pack.GetNBlocks();
      auto ib = md->GetBoundsI(IndexDomain::interior, TE::E3);
      auto jb = md->GetBoundsJ(IndexDomain::interior, TE::E3);
      auto kb = md->GetBoundsK(IndexDomain::interior, TE::E3);

      par_for(
          PARTHENON_AUTO_LABEL, 0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            pack.flux(b, TE::E3, MAG(), k, j, i) = GetEdgeEMF<TE::E3, emf_averaging>(
                SubPack<Axis::IAXIS, Axis::JAXIS>(pack, b, k, j, i));
            if (ndim > 2) {
              pack.flux(b, TE::E1, MAG(), k, j, i) = GetEdgeEMF<TE::E1, emf_averaging>(
                  SubPack<Axis::JAXIS, Axis::KAXIS>(pack, b, k, j, i));
              pack.flux(b, TE::E2, MAG(), k, j, i) = GetEdgeEMF<TE::E2, emf_averaging>(
                  SubPack<Axis::KAXIS, Axis::IAXIS>(pack, b, k, j, i));
            }
          });
    }
    return TaskStatus::complete;
  }
};

TaskID AddFluxTasks(TaskID prev, TaskList &tl, MeshData *md) {
  // calculate fluxes -- CalculateFluxes

  // needs to return task id from last task
  TaskID get_fluxes = prev;
  auto cfg = GetConfig(md);
  if (cfg->Get<ReconstructionStrategy>() == ReconstructionStrategy::scratchpad) {
    // --8<-- [start:add_task]
    get_fluxes = tl.AddTask(
        prev, "hydro::CalculateFluxes",
        [](MeshData *md, Config *cfg) {
          return Dispatcher<CalculateFluxesNested>(PARTHENON_AUTO_LABEL, cfg).execute(md);
        },
        md, cfg.get());
    // --8<-- [end:add_task]
  } else {
    get_fluxes = tl.AddTask(
        prev, "hydro::CalculateFluxes",
        [](MeshData *md, Config *cfg) {
          return Dispatcher<CalculateFluxesScratch>(PARTHENON_AUTO_LABEL, cfg)
              .execute(md);
        },
        md, cfg.get());
  }
  auto get_emf = tl.AddTask(
      get_fluxes, "hydro::CalculateEMF",
      [](MeshData *md, Config *cfg) {
        return Dispatcher<CalculateEMF>(PARTHENON_AUTO_LABEL, cfg).execute(md);
      },
      md, cfg.get());

  return get_emf;
}
}  // namespace kamayan::hydro
