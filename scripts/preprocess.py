# template_cli.py
import argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DESCRIPTION")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p

def main():
    args = build_parser().parse_args()
    # TODO: implement step
    # read args.input, write args.output

if __name__ == "__main__":
    main()
