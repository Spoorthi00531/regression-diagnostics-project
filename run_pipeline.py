"""
run_pipeline.py - run the WHOLE analysis from the command line.

    python run_pipeline.py

It loads the data, fits the models, runs every diagnostic, saves all plots and
CSV files into outputs/, and writes reports/project_results.md with the real numbers.
(The notebook and the dashboard use the exact same functions from sml_core.py.)
"""
import os
import matplotlib
matplotlib.use("Agg")            # draw to files; no pop-up windows needed

import sml_core as core


def main():
    print("Loading data and running the full analysis ...")
    res = core.run_analysis(rule="cook")
    files = core.save_all_outputs(res)

    md = core.build_results_markdown(res)
    path = os.path.join(core.REPORT_DIR, "project_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n=== KEY RESULTS ===")
    print("Test metrics (original):\n", res["comp"]["test_metrics"].round(4))
    print("\n" + res["text"]["bp"])
    print(res["text"]["jb"])
    print(res["text"]["vif"])
    print("\n" + res["text"]["influence"])
    print("\n" + res["text"]["comparison"])
    print(f"\nSaved {len(files)} files under outputs/ and the report at {path}")


if __name__ == "__main__":
    main()
