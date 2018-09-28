def search(prefixes, getnext, stop, value, K=1):
    """Beam search: maintain a collection of K prefixes, sorted by value."""
    def extend(prefix):
        return [ prefix + [item] for item in getnext(prefix)]
    extended = [   item  for prefix in prefixes for item in extend(prefix) ]
    reranked = list(reversed(sorted(extended, key=value)))
    pruned =   reranked[:K]
    if all([stop(prefix) for prefix in pruned]):
        return pruned
    else:
        return search(pruned, getnext, stop, value, K)

# Some toy example    

def value(prefix):
    return sum(val for val, _ in prefix)

def stop(prefix):
    return prefix[-1] == 'z' or len(prefix) > 10 


def getnext(prefix):
    _, last = prefix[-1]
    return [(1, chr(ord(last)+1)), (0.1, chr(ord(last)+2))]

result = search([[(1,'a')]], getnext=getnext, stop=stop, value=value, K=5)

for r in result:
    print(''.join([letter for _, letter in r]))

    
