#include "search_engine.h"

#include "evaluation_context.h"
#include "evaluator.h"

#include "algorithms/ordered_set.h"
#include "plugins/plugin.h"
#include "task_utils/successor_generator.h"
#include "task_utils/task_properties.h"
#include "tasks/root_task.h"
#include "utils/countdown_timer.h"
#include "utils/rng_options.h"
#include "utils/system.h"
#include "utils/timer.h"

#include <cassert>
#include <iostream>
#include <limits>

using namespace std;
using utils::ExitCode;

class PruningMethod;

successor_generator::SuccessorGenerator &get_successor_generator(
    const TaskProxy &task_proxy, utils::LogProxy &log) {
    log << "Building successor generator..." << flush;
    int peak_memory_before = utils::get_peak_memory_in_kb();
    utils::Timer successor_generator_timer;
    successor_generator::SuccessorGenerator &successor_generator =
        successor_generator::g_successor_generators[task_proxy];
    successor_generator_timer.stop();
    log << "done!" << endl;
    int peak_memory_after = utils::get_peak_memory_in_kb();
    int memory_diff = peak_memory_after - peak_memory_before;
    log << "peak memory difference for successor generator creation: "
        << memory_diff << " KB" << endl
        << "time for successor generation creation: "
        << successor_generator_timer << endl;
    return successor_generator;
}

SearchEngine::SearchEngine(const plugins::Options &opts)
    : description(opts.get_unparsed_config()),
      status(IN_PROGRESS),
      solution_found(false),
      task(tasks::g_root_task),
      task_proxy(*task),
      log(utils::get_log_from_options(opts)),
      state_registry(task_proxy),
      successor_generator(get_successor_generator(task_proxy, log)),
      search_space(state_registry, log),
      statistics(log),
      cost_type(opts.get<OperatorCost>("cost_type")),
      is_unit_cost(task_properties::is_unit_cost(task_proxy)),
      max_time(opts.get<double>("max_time")),
      search_dump_id(opts.get<int>("search_dump_id"))  {
    if (opts.get<int>("bound") < 0) {
        cerr << "error: negative cost bound " << opts.get<int>("bound") << endl;
        utils::exit_with(ExitCode::SEARCH_INPUT_ERROR);
    }
    bound = opts.get<int>("bound");
    task_properties::print_variable_statistics(task_proxy);
    if (search_dump_id > 0) {
        string search_dump_filename = "search_dump_" + std::to_string(search_dump_id) + ".txt";
        search_dump_ofstream.open(search_dump_filename, ios::out | ios::trunc);
        search_dump_ofstream << "id\tN\tg\th\tf\tb\th0\thmin\tN_of_hmin\tf_max\tpath" << endl;
        cout << "Dumping search into file: " << search_dump_filename << endl;
    }
}

SearchEngine::~SearchEngine() {
    if (search_dump_id > 0) {
        search_dump_ofstream.close();
    }
}

bool SearchEngine::found_solution() const {
    return solution_found;
}

SearchStatus SearchEngine::get_status() const {
    return status;
}

const Plan &SearchEngine::get_plan() const {
    assert(solution_found);
    return plan;
}

void SearchEngine::set_plan(const Plan &p) {
    solution_found = true;
    plan = p;
}

void SearchEngine::search() {    
    initialize();
    utils::CountdownTimer timer(max_time);
    while (status == IN_PROGRESS) {
        status = step();
        if (timer.is_expired()) {
            log << "Time limit reached. Abort search." << endl;
            status = TIMEOUT;
            break;
        }
    }
    // TODO: Revise when and which search times are logged.
    log << "Actual search time: " << timer.get_elapsed_time() << endl;
}

bool SearchEngine::check_goal_and_set_plan(const State &state) {
    if (task_properties::is_goal_state(task_proxy, state)) {
        log << "Solution found!" << endl;
        Plan plan;
        search_space.trace_path(state, plan);
        set_plan(plan);
        return true;
    }
    return false;
}

void SearchEngine::save_plan_if_necessary() {
    if (found_solution()) {
        plan_manager.save_plan(get_plan(), task_proxy);
    }
}

int SearchEngine::get_adjusted_cost(const OperatorProxy &op) const {
    return get_adjusted_action_cost(op, cost_type, is_unit_cost);
}

/* TODO: merge this into add_options_to_feature when all search
         engines support pruning.

   Method doesn't belong here because it's only useful for certain derived classes.
   TODO: Figure out where it belongs and move it there. */
void SearchEngine::add_pruning_option(plugins::Feature &feature) {
    feature.add_option<shared_ptr<PruningMethod>>(
        "pruning",
        "Pruning methods can prune or reorder the set of applicable operators in "
        "each state and thereby influence the number and order of successor states "
        "that are considered.",
        "null()");
}

void SearchEngine::add_options_to_feature(plugins::Feature &feature) {
    ::add_cost_type_option_to_feature(feature);
    feature.add_option<int>(
        "bound",
        "exclusive depth bound on g-values. Cutoffs are always performed according to "
        "the real cost, regardless of the cost_type parameter", "infinity");
    feature.add_option<double>(
        "max_time",
        "maximum time in seconds the search is allowed to run for. The "
        "timeout is only checked after each complete search step "
        "(usually a node expansion), so the actual runtime can be arbitrarily "
        "longer. Therefore, this parameter should not be used for time-limiting "
        "experiments. Timed-out searches are treated as failed searches, "
        "just like incomplete search algorithms that exhaust their search space.",
        "infinity");
    feature.add_option<int>(
        "search_dump_id",
        "if value > 0, dump search info into a file called 'search_dump_<value>.txt'", "0");
    utils::add_log_options_to_feature(feature);
}

/* Method doesn't belong here because it's only useful for certain derived classes.
   TODO: Figure out where it belongs and move it there. */
void SearchEngine::add_succ_order_options(plugins::Feature &feature) {
    vector<string> options;
    feature.add_option<bool>(
        "randomize_successors",
        "randomize the order in which successors are generated",
        "false");
    feature.add_option<bool>(
        "preferred_successors_first",
        "consider preferred operators first",
        "false");
    feature.document_note(
        "Successor ordering",
        "When using randomize_successors=true and "
        "preferred_successors_first=true, randomization happens before "
        "preferred operators are moved to the front.");
    utils::add_rng_options(feature);
}

void print_initial_evaluator_values(
    const EvaluationContext &eval_context) {
    eval_context.get_cache().for_each_evaluator_result(
        [] (const Evaluator *eval, const EvaluationResult &result) {
            if (eval->is_used_for_reporting_minima()) {
                eval->report_value_for_initial_state(result);
            }
        }
        );
}

static class SearchEngineCategoryPlugin : public plugins::TypedCategoryPlugin<SearchEngine> {
public:
    SearchEngineCategoryPlugin() : TypedCategoryPlugin("SearchEngine") {
        // TODO: Replace add synopsis for the wiki page.
        // document_synopsis("...");
    }
}
_category_plugin;

void collect_preferred_operators(
    EvaluationContext &eval_context,
    Evaluator *preferred_operator_evaluator,
    ordered_set::OrderedSet<OperatorID> &preferred_operators) {
    if (!eval_context.is_evaluator_value_infinite(preferred_operator_evaluator)) {
        for (OperatorID op_id : eval_context.get_preferred_operators(preferred_operator_evaluator)) {
            preferred_operators.insert(op_id);
        }
    }
}
