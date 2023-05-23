#! /usr/bin/env python3

"""Solve some tasks with A* and the LM-Cut heuristic."""

import os
import os.path
import copy

from downward.experiment import FastDownwardExperiment
from downward.reports.absolute import AbsoluteReport, PlanningReport
from downward.reports.scatter import ScatterPlotReport
from lab.environments import BaselSlurmEnvironment, LocalEnvironment
from downward.experiment import FastDownwardRun

ATTRIBUTES = ["coverage", "error", "expansions", "planner_memory", "planner_time"]

SUITE = ["depot:p01.pddl", "gripper:prob01.pddl", "mystery:prob07.pddl"]
ENV = LocalEnvironment(processes=2)
# Use path to your Fast Downward repository.
REPO = os.environ["DOWNWARD_REPO"]
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
# If REVISION_CACHE is None, the default ./data/revision-cache is used.
REVISION_CACHE = os.environ.get("DOWNWARD_REVISION_CACHE")
REV = "main"

exp = FastDownwardExperiment(environment=ENV, revision_cache=REVISION_CACHE)

# Add built-in parsers to the experiment.
exp.add_parser(exp.EXITCODE_PARSER)
exp.add_parser(exp.TRANSLATOR_PARSER)
exp.add_parser(exp.SINGLE_SEARCH_PARSER)
exp.add_parser(exp.PLANNER_PARSER)


heuristics = {
    "ff" : "ff()",
    "lmcut" : "lmcut()"
}

exp.add_suite(BENCHMARKS_DIR, SUITE)
for hname, heuristic in heuristics.items():
    exp.add_algorithm("gbfs_" + hname, REPO, REV, ["--evaluator", "h=" + heuristic, "--search",  "eager_greedy([h],search_dump_id=1)"])
    exp.add_algorithm("astar_" + hname, REPO, REV, ["--evaluator", "h=" + heuristic, "--search", "astar(h,search_dump_id=1)"])

# Add step that writes experiment files to disk.
exp.add_step("build", exp.build)

# Add step that executes all runs.
exp.add_step("start", exp.start_runs)

def collect_data():
    print(exp)

# Add step that collects properties from run directories and
# writes them to *-eval/properties.
exp.add_fetcher(name="fetch")




class SearchDumpReport(PlanningReport):
    def get_text(self):
        data = [ ("search_algorithm", "heuristic", "domain", "problem", "expansions", "search_dump_file") ]
        for (domain, problem), runs in self.problem_runs.items():
            for run in runs:
                algorithm = run["algorithm"]
                search_algo = algorithm.split("_")[0]
                heuristic = algorithm.split("_")[1]
                domain = run["domain"]
                run_dir = run["run_dir"]
                data.append( (search_algo, heuristic, domain, run["problem"], str(run["expansions"]), os.path.join(exp.path, run_dir, "search_dump_1.txt") ))
        return "\n".join(map(lambda line: ",".join(line),data))

def my_filter(run):
    return run["coverage"] == 1

# Add report step (AbsoluteReport is the standard report).
#exp.add_report(
#    PlanningReport(attributes=["expansions", "coverage"], format="txt", filter=my_filter), outfile="report.html"
#)

exp.add_report(
    SearchDumpReport(filter=my_filter),outfile="data.csv"
)


# Parse the commandline and show or run experiment steps.
exp.run_steps()
