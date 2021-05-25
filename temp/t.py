class A:
    def __init__(self):
        self.f()

    def f(self):
        print("AAAA")


class B(A):
    def __init__(self):
        super(B, self).__init__()

    def f(self):
        print("BBBB")


if __name__ == "__main__":
    b = B()