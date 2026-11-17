import argparse
import cascade as m


def main():
    ap = argparse.ArgumentParser(description="级联阴影分割")
        ap.add_argument("--levels", type=int, default=4)
        ap.add_argument("--out", required=True)
    a = ap.parse_args()
        print("TODO")


if __name__ == "__main__":
    main()
