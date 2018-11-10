# IMPORTANT NOTE: If you get an error unpickling these files in train and test then run this!


def run(orig, dest):
    content = ''
    outsize = 0
    with open(orig, 'rb') as infile:
        content = infile.read()
    with open(dest, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

    print("Done. Saved %s bytes." % (len(content) - outsize))


original = "data/movie_db.pkl"
destination = "data/movie_db.pkl"
run(original, destination)

original = "data/movie_dict.pkl"
destination = "data/movie_dict.pkl"
run(original, destination)

original = "data/movie_user_goals.pkl"
destination = "data/movie_user_goals.pkl"
run(original, destination)