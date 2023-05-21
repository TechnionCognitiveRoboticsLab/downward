#! /usr/bin/env python3

"""Solve some tasks with A* and the LM-Cut heuristic."""

import os
import os.path
import copy

from downward.experiment import FastDownwardExperiment
from downward.reports.absolute import AbsoluteReport
from downward.reports.scatter import ScatterPlotReport
from lab.environments import BaselSlurmEnvironment, LocalEnvironment
from downward.experiment import FastDownwardRun


class FastDownwardExperimentWithSerialNumber(FastDownwardExperiment):
    def __init__(self, path=None, environment=None, revision_cache=None):
        super().__init__(path, environment, revision_cache)

    def _add_runs(self):
        i = 0
        tasks = self._get_tasks()
        for algo in self._algorithms.values():
            for task in tasks:        
                i = i + 1
                new_algo = copy.deepcopy(algo)        
                new_algo.component_options = list(map(lambda x: x.replace("%SN%", str(i)), algo.component_options))
                print(self, new_algo, task)
                self.add_run(FastDownwardRun(self, new_algo, task))



ATTRIBUTES = ["coverage", "error", "expansions", "planner_memory", "planner_time"]

SUITE = ["depot:p01.pddl", "gripper:prob01.pddl", "mystery:prob07.pddl"]
ENV = LocalEnvironment(processes=2)
# Use path to your Fast Downward repository.
REPO = os.environ["DOWNWARD_REPO"]
BENCHMARKS_DIR = os.environ["DOWNWARD_BENCHMARKS"]
# If REVISION_CACHE is None, the default ./data/revision-cache is used.
REVISION_CACHE = os.environ.get("DOWNWARD_REVISION_CACHE")
REV = "main"

exp = FastDownwardExperimentWithSerialNumber(environment=ENV, revision_cache=REVISION_CACHE)

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
    exp.add_algorithm("gbfs_" + hname, REPO, REV, ["--evaluator", "h=" + heuristic, "--search",  "eager_greedy([h],search_dump_id=%SN%)"])
    exp.add_algorithm("astar_" + hname, REPO, REV, ["--evaluator", "h=" + heuristic, "--search", "astar(h,search_dump_id=%SN%)"])

# Add step that writes experiment files to disk.
exp.add_step("build", exp.build)

# Add step that executes all runs.
exp.add_step("start", exp.start_runs)

# Add step that collects properties from run directories and
# writes them to *-eval/properties.
exp.add_fetcher(name="fetch")

# Add report step (AbsoluteReport is the standard report).
exp.add_report(
    AbsoluteReport(attributes=ATTRIBUTES, format="html"), outfile="report.html"
)

# Add scatter plot report step.
exp.add_report(
    ScatterPlotReport(attributes=["expansions"], filter_algorithm=["blind", "lmcut"]),
    outfile="scatterplot.png",
)

# Parse the commandline and show or run experiment steps.
exp.run_steps()