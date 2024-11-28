import gc
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from .benchmarks.interface import BenchmarkCreator
from .param_handling import BenchmarkConfig
from .plot_results import plot_benchmark_result_table

LOGGER = logging.getLogger(__name__)


def run_benchmarks(
    benchmark_config: BenchmarkConfig,
    benchmark_creator: BenchmarkCreator,
    param_prefix: str = "P--",
    additional_param_name_short: bool = True,
    runtime_prefix: str = "R--",
    memory_prefix: str = "M--",
) -> pd.DataFrame:
    """Runs the different kernel configurations and summarizes the results in a DataFrame.

    Args:
        benchmark_config: The benchmark configuration.
        param_prefix: The prefix to add to the parameter names in the DataFrame.
    """
    param_dicts = benchmark_config.get_param_dicts()
    kernel_specs = benchmark_config.kernel_specs
    results = []
    for i, param_dict in enumerate(param_dicts):
        LOGGER.info(f"Parameter combination ({i+1}/{len(param_dicts)}): {param_dict}")
        # add a prefix to easily identify the parameters in the DataFrame
        result_dict = {f"{param_prefix}{k}": v for k, v in param_dict.items()}
        for k, kernel_spec in enumerate(kernel_specs):
            benchmark = benchmark_creator(kernel_spec, param_dict)
            benchmark.set_params(kernel_spec.additional_params)
            benchmark.setup_benchmark()
            runtime_results = benchmark.run_benchmark()
            result_dict[
                f"{runtime_prefix}{kernel_spec.to_string(short_param_name=additional_param_name_short)}"
            ] = runtime_results.runtime
            result_dict[
                f"{memory_prefix}{kernel_spec.to_string(short_param_name=additional_param_name_short)}"
            ] = runtime_results.peak_memory_allocated
            LOGGER.info(
                (
                    f"Kernel ({k+1}/{len(kernel_specs)}): {kernel_spec.to_string()} finished.",
                    f" Runtime: {runtime_results.runtime} ms. Peak memory: {float(runtime_results.peak_memory_allocated / 10**9)} GB.",
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result_dict)

    return pd.DataFrame(results)


def run_and_record_benchmarks(
    benchmark_config: BenchmarkConfig,
    benchmark_creator: BenchmarkCreator,
    output_folder: Path,
):
    import logging

    import tabulate

    LOGGER = logging.getLogger(__name__)

    LOGGER.info(f"Running benchmark: {benchmark_config.benchmark_name}")

    benchmark_folder = output_folder / benchmark_config.benchmark_name
    benchmark_folder.mkdir(parents=True, exist_ok=False)

    OmegaConf.save(
        OmegaConf.create(asdict(benchmark_config)), benchmark_folder / "config.yaml"
    )

    result_df = run_benchmarks(
        benchmark_config=benchmark_config,
        benchmark_creator=benchmark_creator,
        additional_param_name_short=True,
    )

    LOGGER.info(
        f"Results:\n{tabulate.tabulate(result_df, headers='keys', tablefmt='pretty')}"
    )
    LOGGER.info(f"Saving results to {benchmark_folder}")
    result_df.to_csv(benchmark_folder / "results.csv")

    def plot_result_table(
        additional_exclude_col_regex: str, plot_name_suffix: str, y_label: str
    ):
        fig = plot_benchmark_result_table(
            result_df,
            benchmark_config.x_axis_param,
            title=f"Runtime--{benchmark_config.get_plot_title()}",
            additional_exclude_col_regex=additional_exclude_col_regex,
            y_label=y_label,
        )
        fig.savefig(
            benchmark_folder
            / f"plot_{benchmark_config.benchmark_name}_{plot_name_suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            benchmark_folder
            / f"plot_{benchmark_config.benchmark_name}_{plot_name_suffix}.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            benchmark_folder
            / f"plot_{benchmark_config.benchmark_name}_{plot_name_suffix}.svg",
            bbox_inches="tight",
        )

    # runtime plot
    plot_result_table("M--.*", "runtime", "Time [ms]")
    # memory plot
    plot_result_table("R--.*", "memory", "Memory [bytes]")
