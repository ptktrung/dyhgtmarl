"""Run every table script. Scripts with missing inputs report and skip."""
import pathlib, runpy, sys, traceback

HERE = pathlib.Path(__file__).resolve().parent / "tables"
sys.path.insert(0, str(HERE))

def main():
    for p in sorted(HERE.glob("table_*.py")):
        print(f"--- {p.name}")
        try:
            runpy.run_path(str(p), run_name="__main__")
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    main()
