
if __name__ == "__main__":
    with open("4d1000.txt") as f1:
        for line in f1:
            a, b, c, d = line.split()
            print(f"{a} {b}")

