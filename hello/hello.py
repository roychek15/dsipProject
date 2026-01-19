import argparse


def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, default = "World", help="Name to greet")
    parser.add_argument('-v','--version', action="version", help="helloworld 1.0")

    args = parser.parse_args()
    print(hello(args.name, args.version))