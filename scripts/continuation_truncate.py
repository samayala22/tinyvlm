import argparse, pickle
import continuation as cont

def main(args):
    filename = args.filename
    md = cont.load_metadata(filename)

    new_md = md
    new_md.bifurcation_test = md.bifurcation_test[:, :args.iterations]
    for key in md.bifurcation:
        new_md.bifurcation[key] = [idx for idx in md.bifurcation[key] if idx < args.iterations]
    new_md.stable = md.stable[:args.iterations]
    new_md.X = md.X[:, :args.iterations]
    new_md.floquet_exponents = md.floquet_exponents[:, :args.iterations]

    new_filename = cont.metadata_filename(new_md)

    with open(f"build/{new_filename}", 'wb') as f:
        pickle.dump(new_md, f)
    print(f"Truncated metadata saved to build/{new_filename}")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="A filename")
    parser.add_argument("iterations", type=int, help="Truncate to this many iterations")
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    main(args)