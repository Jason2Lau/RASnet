import argparse
import os
from DAI.daimodel import DAIModel


def parse_args():
    parser = argparse.ArgumentParser(description="remove shadow pattern, high quality, detailed")
    parser.add_argument(
        "--pretrained_dai",
        type=str,
        default="weights/sd",
    )
    parser.add_argument("--controlnet", type=str, default=None)
    parser.add_argument("--cross_vae", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=960)

    parser.add_argument(
        "--input_dir", type=str, required=True, help="input image derectory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="output image derectory."
    )
    parser.add_argument(
        "--concat_dir",
        type=str,
        default=None,
        help="concat input and output image derectory.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    daimodel = DAIModel(args=args, mode="inference")
    if args.output_dir is None:
        result_dir = os.path.join(
            "outputs", "results", os.path.basename(args.input_dir)
        )
    else:
        result_dir = args.output_dir
    if args.concat_dir is None:
        concat_dir = os.path.join(
            "outputs", "concats", os.path.basename(args.input_dir)
        )
    else:
        concat_dir = args.concat_dir
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(concat_dir, exist_ok=True)
    daimodel.inference(args.input_dir, result_dir, concat_dir, args.input_size)


if __name__ == "__main__":
    main()
