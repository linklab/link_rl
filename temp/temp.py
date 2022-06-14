def recursive_seq(n):  # 재귀적 방법
    if n == 2: print("!")
    if n <= 2:
        return n-1
    return recursive_seq(n-1) + recursive_seq(n-2) + 3


def iterative_seq(n):  # 반복적 방법
    s = [0]
    for i in range(1, n+1):
        s.append(0)

    for i in range(1, n+1):
        if i <= 2:
            s[i] = i-1
        else:
            print("@")
            s[i] = s[i-1] + s[i-2] + 3

    return s[n]


if __name__ == "__main__":
    print(recursive_seq(6))
    print(iterative_seq(6))


