def generate_random_prime(bits, primality_test):
    """Generate random prime number with n bits."""
    get_random_t = lambda: random.getrandbits(bits) | 1 << bits | 1
    p = get_random_t()
    for i in itertools.count(1):
        if primality_test(p):
            return p
        else:
            if i % (bits * 2) == 0:
                p = get_random_t()
            else:
                p += 2  # Add 2 since we are only interested in odd numbers

                @logged("%b %d %Y - %H:%M:%S")
                def simple_is_prime(n):
                    """Returns True if n is a prime. False otherwise."""
                    if n % 2 == 0:
                        return n == 2
                    if n % 3 == 0:
                        return n == 3
                    k = 6L
                    while (k - 1) ** 2 <= n:
                        if n % (k - 1) == 0 or n % (k + 1) == 0:
                            return False
                        k += 6
                    return True


                    def rabin_miller_is_prime(n, k=20):
                        """
                        Test n for primality using Rabin-Miller algorithm, with k
                        random witness attempts. False return means n is certainly a composite.
                        True return value indicates n is *probably* a prime. False positive
                        probability is reduced exponentially the larger k gets.
                        """
                        b = basic_is_prime(n, K=100)
                        if b is not None:
                            return b

                        m = n - 1
                        s = 0
                        while m % 2 == 0:
                            s += 1
                            m //= 2
                        liars = set()
                        get_new_x = lambda: random.randint(2, n - 1)
                        while len(liars) < k:
                            x = get_new_x()
                            while x in liars:
                                x = get_new_x()
                            xi = pow(x, m, n)
                            witness = True
                            if xi == 1 or xi == n - 1:
                                witness = False
                            else:
                                for __ in xrange(s - 1):
                                    xi = (xi ** 2) % n
                                    if xi == 1:
                                        return False
                                    elif xi == n - 1:
                                        witness = False
                                        break
                                xi = (xi ** 2) % n
                                if xi != 1:
                                    return False
                            if witness:
                                return False
                            else:
                                liars.add(x)
                        return True

 def basic_is_prime(n, K=-1):
    """Returns True if n is a prime, and False it is a composite
    by trying to divide it by two and first K odd primes. Returns
    None if test is inconclusive."""
    if n % 2 == 0:
        return n == 2
    for p in primes_list.less_than_hundred_thousand[:K]:
        if n % p == 0:
            return n == p
 return None


 def power(x, m, n):
    """Calculate x^m modulo n using O(log(m)) operations."""
    a = 1
    while m > 0:
        if m % 2 == 1:
            a = (a * x) % n
        x = (x * x) % n
        m //= 2
    return a

    def extended_gcd(a, b):
        """Returns pair (x, y) such that xa + yb = gcd(a, b)"""
        x, lastx, y, lasty = 0, 1, 1, 0
        while b != 0:
            q, r = divmod(a, b)
            a, b = b, r
            x, lastx = lastx - q * x, x
            y, lasty = lasty - q * y, y
        return lastx, lasty


    def multiplicative_inverse(e, n):
        """Find the multiplicative inverse of e mod n."""
        x, y = extended_gcd(e, n)
        if x < 0:
            return n + x
        return x


    def rsa_generate_key(bits):
        p = generate_random_prime(bits / 2)
        q = generate_random_prime(bits / 2)
        # Ensure q != p, though for large values of bits this is
        # statistically very unlikely
        while q == p:
            q = generate_random_prime(bits / 2)
        n = p * q
        phi = (p - 1) * (q - 1)
        # Here we pick a random e, but a fixed value for e can also be used.
        while True:
            e = random.randint(3, phi - 1)
            if fractions.gcd(e, phi) == 1:
                break
        d = multiplicative_inverse(e, phi)
        return (n, e, d)


    def rsa_encrypt(message, n, e):
        return modular.power(message, e, n)


    def rsa_decrypt(cipher, n, d):
        return modular.power(cipher, d, n)
