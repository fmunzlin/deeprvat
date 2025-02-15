import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from deeprvat.preprocessing.preprocess import cli as preprocess_cli
from click.testing import CliRunner
from pathlib import Path
import h5py
import pytest

script_dir = Path(__file__).resolve().parent
tests_data_dir = script_dir / "test_data/preprocessing"


def load_h5_archive(h5_path):
    with h5py.File(h5_path, "r") as f:
        written_samples = f["samples"][:]
        written_variant_matrix = f["variant_matrix"][:]
        written_genotype_matrix = f["genotype_matrix"][:]

        return written_samples, written_variant_matrix, written_genotype_matrix


@pytest.mark.parametrize(
    "test_data_name_dir, extra_cli_params, genotype_file_name",
    [
        (
            "no_filters_minimal",
            [
                "--chromosomes",
                "1",
            ],
            "genotypes_chr1.h5",
        ),
        (
            "filter_variants_minimal",
            [
                "--chromosomes",
                "1",
                "--exclude-variants",
                f"{(tests_data_dir / 'process_sparse_gt/filter_variants_minimal/input/qc').as_posix()}",
            ],
            "genotypes_chr1.h5",
        ),
        (
            "filter_samples_minimal",
            [
                "--chromosomes",
                "1",
                "--exclude-samples",
                f"{(tests_data_dir / 'process_sparse_gt/filter_samples_minimal/input/qc').as_posix()}",
            ],
            "genotypes_chr1.h5",
        ),
        (
            "filter_calls_minimal",
            [
                "--chromosomes",
                "1",
                "--exclude-calls",
                f"{(tests_data_dir / 'process_sparse_gt/filter_calls_minimal/input/qc').as_posix()}",
            ],
            "genotypes_chr1.h5",
        ),
        (
            "filter_calls_vars_samples_minimal",
            [
                "--chromosomes",
                "1",
                "--exclude-calls",
                f"{(tests_data_dir / 'process_sparse_gt/filter_calls_vars_samples_minimal/input/qc/calls/').as_posix()}",
                "--exclude-samples",
                f"{(tests_data_dir / 'process_sparse_gt/filter_calls_vars_samples_minimal/input/qc/samples/').as_posix()}",
                "--exclude-variants",
                f"{(tests_data_dir / 'process_sparse_gt/filter_calls_vars_samples_minimal/input/qc/variants/').as_posix()}",
            ],
            "genotypes_chr1.h5",
        ),
    ],
)
def test_process_sparse_gt_file(
    test_data_name_dir, extra_cli_params, genotype_file_name, tmp_path
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "process_sparse_gt" / test_data_name_dir

    test_data_input_dir = current_test_data_dir / "input"

    variant_file = test_data_input_dir / "variants.tsv.gz"
    samples_file = test_data_input_dir / "samples_chr.csv"
    sparse_gt_dir = test_data_input_dir / "sparse_gt"

    preprocessed_dir = tmp_path / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    out_file_base = preprocessed_dir / "genotypes"
    expected_array_archive = current_test_data_dir / "expected/expected_data.npz"

    cli_parameters = [
        "process-sparse-gt",
        *extra_cli_params,
        variant_file.as_posix(),
        samples_file.as_posix(),
        sparse_gt_dir.as_posix(),
        out_file_base.as_posix(),
    ]

    result = cli_runner.invoke(preprocess_cli, cli_parameters)
    assert result.exit_code == 0

    h5_file = out_file_base.as_posix().replace("genotypes", genotype_file_name)

    written_samples, written_variant_matrix, written_genotype_matrix = load_h5_archive(
        h5_path=h5_file
    )

    expected_data = np.load(expected_array_archive.as_posix(), allow_pickle=True)

    assert np.array_equal(written_variant_matrix, expected_data["variant_matrix"])
    assert np.array_equal(written_genotype_matrix, expected_data["genotype_matrix"])
    assert np.array_equal(written_samples, expected_data["samples"])


@pytest.mark.parametrize(
    "test_data_name_dir, input_h5, result_h5",
    [
        (
            "combine_chr1_chr2",
            [
                "genotypes_chr1.h5",
                "genotypes_chr2.h5",
            ],
            "genotypes.h5",
        ),
    ],
)
def test_combine_genotypes(test_data_name_dir, input_h5, result_h5, tmp_path):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "combine_genotypes" / test_data_name_dir

    test_data_input_dir = current_test_data_dir / "input"

    preprocessed_dir = tmp_path / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    combined_output_h5 = preprocessed_dir / result_h5
    expected_array_archive = current_test_data_dir / "expected/expected_data.npz"

    cli_parameters = [
        "combine-genotypes",
        *[(test_data_input_dir / h5f).as_posix() for h5f in input_h5],
        combined_output_h5.as_posix(),
    ]

    result = cli_runner.invoke(preprocess_cli, cli_parameters)
    assert result.exit_code == 0

    written_samples, written_variant_matrix, written_genotype_matrix = load_h5_archive(
        h5_path=combined_output_h5
    )

    expected_data = np.load(expected_array_archive.as_posix(), allow_pickle=True)

    assert np.array_equal(written_variant_matrix, expected_data["variant_matrix"])
    assert np.array_equal(written_genotype_matrix, expected_data["genotype_matrix"])
    assert np.array_equal(written_samples, expected_data["samples"])


@pytest.mark.parametrize(
    "test_data_name_dir, input_variants, output_variants, output_duplicates",
    [
        (
            "add_variant_ids_tsv",
            "variants_no_id.tsv.gz",
            "variants.tsv.gz",
            "duplicates.tsv",
        ),
        (
            "add_variant_ids_parquet",
            "variants_no_id.tsv.gz",
            "variants.parquet",
            "duplicates.parquet",
        ),
    ],
)
def test_add_variant_ids(
    test_data_name_dir, input_variants, output_variants, output_duplicates, tmp_path
):
    cli_runner = CliRunner()

    current_test_data_dir = tests_data_dir / "add_variant_ids" / test_data_name_dir

    test_data_input_dir = current_test_data_dir / "input"

    expected_variants_file = current_test_data_dir / "expected/expected_variants.tsv.gz"

    norm_variants_dir = tmp_path / "norm/variants"
    qc_duplicate_vars_dir = tmp_path / "qc/duplicate_vars"
    norm_variants_dir.mkdir(parents=True, exist_ok=True)
    qc_duplicate_vars_dir.mkdir(parents=True, exist_ok=True)

    cli_parameters = [
        "add-variant-ids",
        (test_data_input_dir / input_variants).as_posix(),
        (norm_variants_dir / output_variants).as_posix(),
        (qc_duplicate_vars_dir / output_duplicates).as_posix(),
    ]

    result = cli_runner.invoke(preprocess_cli, cli_parameters)
    assert result.exit_code == 0

    written_variants_path = norm_variants_dir / output_variants

    if written_variants_path.suffix == ".parquet":
        written_variants = pd.read_parquet(written_variants_path)
    else:
        written_variants = pd.read_csv(written_variants_path, sep="\t")

    expected_variants = pd.read_csv(expected_variants_file, sep="\t")

    assert_frame_equal(written_variants, expected_variants)


@pytest.mark.parametrize(
    "test_data_name_dir, extra_cli_params, input_h5, result_h5",
    [
        (
            "no_filters_minimal_split",
            [
                "--chromosomes",
                "1,2",
            ],
            [
                "genotypes_chr1.h5",
                "genotypes_chr2.h5",
            ],
            "genotypes.h5",
        ),
        (
            "filter_calls_variants_samples_minimal_split",
            [
                "--chromosomes",
                "1,2",
                "--exclude-calls",
                f"{(tests_data_dir / 'process_and_combine_sparse_gt/filter_calls_variants_samples_minimal_split/input/qc/calls/').as_posix()}",
                "--exclude-samples",
                f"{(tests_data_dir / 'process_and_combine_sparse_gt/filter_calls_variants_samples_minimal_split/input/qc/samples/').as_posix()}",
                "--exclude-variants",
                f"{(tests_data_dir / 'process_and_combine_sparse_gt/filter_calls_variants_samples_minimal_split/input/qc/variants/').as_posix()}",
            ],
            [
                "genotypes_chr1.h5",
                "genotypes_chr2.h5",
            ],
            "genotypes.h5",
        ),
    ],
)
def test_process_and_combine_sparse_gt(
    test_data_name_dir, extra_cli_params, input_h5, result_h5, tmp_path
):
    cli_runner = CliRunner()

    current_test_data_dir = (
        tests_data_dir / "process_and_combine_sparse_gt" / test_data_name_dir
    )

    test_data_input_dir = current_test_data_dir / "input"

    variant_file = test_data_input_dir / "variants.tsv.gz"
    samples_file = test_data_input_dir / "samples_chr.csv"
    sparse_gt_dir = test_data_input_dir / "sparse_gt"

    preprocessed_dir = tmp_path / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    out_file_base = preprocessed_dir / "genotypes"
    expected_array_archive = current_test_data_dir / "expected/expected_data.npz"
    combined_output_h5 = preprocessed_dir / result_h5

    cli_parameters_process = [
        "process-sparse-gt",
        *extra_cli_params,
        variant_file.as_posix(),
        samples_file.as_posix(),
        sparse_gt_dir.as_posix(),
        out_file_base.as_posix(),
    ]

    result_process = cli_runner.invoke(preprocess_cli, cli_parameters_process)
    assert result_process.exit_code == 0

    cli_parameters_combine = [
        "combine-genotypes",
        *[(preprocessed_dir / h5f).as_posix() for h5f in input_h5],
        combined_output_h5.as_posix(),
    ]

    result_combine = cli_runner.invoke(preprocess_cli, cli_parameters_combine)
    assert result_combine.exit_code == 0

    written_samples, written_variant_matrix, written_genotype_matrix = load_h5_archive(
        h5_path=combined_output_h5
    )

    expected_data = np.load(expected_array_archive.as_posix(), allow_pickle=True)

    assert np.array_equal(written_variant_matrix, expected_data["variant_matrix"])
    assert np.array_equal(written_genotype_matrix, expected_data["genotype_matrix"])
    assert np.array_equal(written_samples, expected_data["samples"])
