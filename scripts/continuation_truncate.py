import argparse, pickle
import continuation as cont

def main(args):
    filename = args.filename
    md = cont.load_metadata(filename)

    new_md = md
    new_md.bifurcation_test = md.bifurcation_test[:, args.start:args.end]
    for key in md.bifurcation:
        new_md.bifurcation[key] = [idx for idx in md.bifurcation[key] if args.start <= idx < args.end]
    new_md.stable = md.stable[args.start:args.end]
    new_md.X = md.X[:, args.start:args.end]
    new_md.floquet_exponents = md.floquet_exponents[:, args.start:args.end]

    new_filename = cont.metadata_filename(new_md)

    with open(f"build/{new_filename}.pkl", 'wb') as f:
        pickle.dump(new_md, f)
    print(f"Truncated metadata saved to build/{new_filename}")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="A filename")
    parser.add_argument("start", type=int, help="Start from this iteration")
    parser.add_argument("end", type=int, help="End at this iteration")

    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    main(args)