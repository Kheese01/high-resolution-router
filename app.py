# app.py
import argparse
from infer import run

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="output.png")
args = parser.parse_args()

run(args.input, args.output)
