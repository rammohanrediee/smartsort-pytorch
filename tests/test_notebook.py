from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01_smartsort_baseline.ipynb"


def load_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_has_no_execution_errors():
    notebook = load_notebook()
    errors = [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []


def test_all_code_cells_were_executed():
    notebook = load_notebook()
    unexecuted = [
        cell for cell in notebook.cells if cell.cell_type == "code" and cell.execution_count is None
    ]
    assert unexecuted == []
