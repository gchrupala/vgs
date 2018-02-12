def parse_map(lines):
    M = {}
    for line in lines:
        fields  =line.split()
        M[fields[0]] = ' '.join(fields[1:])
    return M

