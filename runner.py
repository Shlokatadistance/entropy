from entropy.shannon_entropy import ShannonEntropy


def main():
    init = ShannonEntropy()
    init.gather_objects().calculate_shannon_entropy()

    print(init.stats.__dict__)


if __name__ == "__main__":
    main()
