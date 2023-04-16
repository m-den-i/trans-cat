def divide_chunks(l, n):
    # looping till length 
    for i in range(0, len(l), n):
        yield l[i:i + n]