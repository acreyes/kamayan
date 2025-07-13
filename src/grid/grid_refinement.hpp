#ifndef GRID_GRID_REFINEMENT_HPP_
#define GRID_GRID_REFINEMENT_HPP_
#include <memory>
#include <string>

#include <amr_criteria/amr_criteria.hpp>

#include "dispatcher/options.hpp"
#include "grid/grid_types.hpp"
#include "grid/scratch_variables.hpp"
#include "kamayan/runtime_parameters.hpp"

namespace kamayan {
POLYMORPHIC_PARM(RefinementCriteria, loehner, first, second);
}

namespace kamayan::grid {
using AmrTag = parthenon::AmrTag;

// --8<-- [start:scratch]
// first-order derivative at centers used in Loehner estimator
using FirstDer_t = ScratchVariable<"firstder", TopologicalType::Cell, 3>;

using RefinementScratch = ScratchVariableList<FirstDer_t>;
using FirstDer = RefinementScratch::type<FirstDer_t>;
// --8<-- [end:scratch]

std::shared_ptr<parthenon::AMRCriteria>
MakeAMRCriteria(const runtime_parameters::RuntimeParameters *rps, std::string block_name);

struct AMRLoehner : public parthenon::AMRCriteria {
  AMRLoehner(const runtime_parameters::RuntimeParameters *rps, std::string &block_name)
      : AMRCriteria(rps->GetPin(), block_name),
        filter(rps->Get<Real>(block_name, "filter")) {}
  void operator()(MeshData *md,
                  parthenon::ParArray1D<AmrTag> &delta_level) const override;

 private:
  Real filter;
};
}  // namespace kamayan::grid
#endif  // GRID_GRID_REFINEMENT_HPP_
