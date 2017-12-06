import numpy as np
import networkx
from random import shuffle, randint

def make_mask(n, kind, axis=0):
    if kind == 'dense':
        a = np.ones((n, n), dtype=np.int32)
    elif kind.startswith('old_ba_'):
        _, _, m = kind.split('_')
        a = old_barabasi_albert(n, int(m))
    elif kind.startswith('ba_'):
        _, m = kind.split('_')
        a = barabasi_albert(n, int(m))
    elif kind.startswith('bae_'): #barabasi-albert with extra nodes
        _, m, e = kind.split('_')
        a = barabasi_albert(n, int(m))
        a = extra(a, n, int(e))
    elif kind.startswith('rande_'): #1D watts-strogatz with extra nodes
        _, m, e = kind.split('_')
        a = watts_strogatz_1d(n, int(m), p=1.)
        a = extra(a, n, int(e))
    elif kind.startswith('ws_'): #1D watts-strogatz with extra nodes
        _, m, pct = kind.split('_')
        a = watts_strogatz_1d(n, int(m)*2, p=float(pct)/100.0)
    elif kind.startswith('br_'): # balanced_random
        _, m = kind.split('_')
        a = balanced_random(n, int(m)*2)
    else:
        raise ValueError('Unknown mask kind: ' + str(kind))
    return a

def barabasi_albert(n, m):
    #print("barabasi_albert", n, m)
    g = networkx.generators.barabasi_albert_graph(n=n, m=m)
    a = networkx.adjacency_matrix(g).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)
    a[0:m,0:m] = 1
    return a

def watts_strogatz_1d(n, k, p):
    assert k % 2 == 0
    g = networkx.generators.random_graphs.watts_strogatz_graph(n, k, p)
    return networkx.adjacency_matrix(g).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)

def extra(a, n, extra):
    # Add extra random edges
    for i in range(extra):
        counts = list(zip(np.sum(a, axis=0), range(n)))
        shuffle(counts)
        counts.sort()
        i = counts[0][1]
        while True:
            j = randint(0, n-1)
            if a[i,j] == 0:
                a[i,j] = 1
                a[j,i] = 1
                break
    return a


# Legacy functions:

def old_barabasi_albert(n, m):
    g = networkx.generators.barabasi_albert_graph(n=n, m=m)
    a = networkx.adjacency_matrix(g).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)
    a[0:m,0:m] = 1
    # add a few more random blocks to match size with watts_strogatz
    target = n * (m*2 + 1)
    while np.sum(a) < target:
        counts = list(zip(np.sum(a, axis=0), range(n)))
        shuffle(counts)
        counts.sort()
        i = counts[0][1]
        while True:
            j = randint(0, n-1)
            if a[i,j] == 0:
                a[i,j] = 1
                a[j,i] = 1
                break
    return a


def watts_strogatz_2d(n, m, p, wrap=True):
    # construct Watts-Strogatz random network on a 2d lattice, having approximately n*m/2 connections

    # get size of lattice & init adjacency matrix
    n0 = int(np.ceil(np.sqrt(n)))
    n1 = n // n0
    assert n0 * n1 == n  # can't construct 2d lattice otherwise
    adjacency_matrix = np.zeros((n0, n1, n0, n1), dtype=np.int8)

    # make nearest neighbor connections
    d = np.square(np.arange(int(np.ceil(np.sqrt(m)))))
    distance_matrix = d.reshape((1, -1)) + d.reshape((-1, 1))
    cutoff = np.sort(distance_matrix.flatten())[m // 2]
    local_connectivity_matrix = distance_matrix <= cutoff
    for i in range(local_connectivity_matrix.shape[0]):
        for j in range(local_connectivity_matrix.shape[1]):
            if local_connectivity_matrix[i, j]:
                if i == 0 and j == 0:
                    pass
                else:
                    if wrap: # should we connect both lattice dimensions end to start?
                        submat0 = adjacency_matrix[:, :, np.mod(np.arange(i, n0+i), n0)]
                        submat1 = submat0[:, :, :, np.mod(np.arange(j, n1+j), n1)]
                        submat1 += np.eye(n0*n1, dtype=np.int8).reshape(n0,n1,n0,n1)
                        submat0[:, :, :, np.mod(np.arange(j, n1 + j), n1)] = submat1
                        adjacency_matrix[:, :, np.mod(np.arange(i, n0 + i), n0)] = submat0
                    else:
                        submat0 = adjacency_matrix[np.arange(n0 - i)]
                        submat1 = submat0[:, np.arange(n1 - j)]
                        submat2 = submat1[:, :, np.arange(i, n0)]
                        submat3 = submat2[:, :, :, np.arange(j, n1)]
                        submat3 += np.eye((n0 - i) * (n1 - j), dtype=np.int8).reshape((n0 - i, n1 - j, n0 - i, n1 - j))
                        submat2[:, :, :, np.arange(j, n1)] = submat3
                        submat1[:, :, np.arange(i, n0)] = submat2
                        submat0[:, np.arange(n1 - j)] = submat1
                        adjacency_matrix[np.arange(n0 - i)] = submat0

    # with probability p rewire each connection to another random end-point
    rewire = np.random.binomial(n=1, p=p, size=(n0, n1, n0, n1))
    rewire_inds = np.nonzero(rewire * adjacency_matrix)

    # remove the rewired connections
    adjacency_matrix[rewire_inds] = 0

    # put back random connections, taking care not to duplicate existing connections
    while True:
        new_end0 = np.random.randint(n0, size=len(rewire_inds[0]))
        new_end1 = np.random.randint(n1, size=len(rewire_inds[0]))
        do_again = [[], []]
        for i in range(len(rewire_inds[0])):
            if adjacency_matrix[rewire_inds[0][i], rewire_inds[1][i], new_end0[i], new_end1[i]] \
                    or rewire_inds[0][i]==new_end0[i] and rewire_inds[1][i]==new_end1[i]:
                do_again[0].append(rewire_inds[0][i])
                do_again[1].append(rewire_inds[1][i])
            else:
                adjacency_matrix[rewire_inds[0][i], rewire_inds[1][i], new_end0[i], new_end1[i]] = 1

        if len(do_again[0]) > 0:
            rewire_inds = do_again
        else:
            break

    # reshape the adjacency matrix back into 2d
    adjacency_matrix = adjacency_matrix.reshape((n, n))
    return adjacency_matrix



def balanced_random(n, m):
    a = np.eye(n, dtype=np.int32)

    cs = list(range(n))
    shuffle(cs)

    # keep track of how many c's are assigned to each k
    kc = [0 for k in range(n)]

    for c in cs:
        # find m eligeble k's but prioritize by low count
        ks = [k for k in range(n) if c != k and kc[k] < m]
        shuffle(ks)
        ks.sort(key=lambda k: kc[k])
        for k in ks[0:m]:
            a[c,k] = 1
            kc[k] += 1

    # ensure each k has m c's
    for k in range(n):
        while kc[k] < m:
            if len(cs) == 0:
                cs = list(range(n))
                shuffle(cs)
            while len(cs) > 0:
                c = cs.pop()
                if a[c,k] == 0:
                    a[c,k] = 1
                    kc[k] += 1
                    break
    return a

# show how much mixing occurs after each internal step
def mix_factor(masks, nsamples=None):

    n = masks[0].shape[0]
    if nsamples is None:
        nsamples = n

    nsamples = min(n, nsamples)
    samples  = list(range(n))
    shuffle(samples)

    masks = [ mask.astype(np.float32) for mask in masks ]

    factors = []
    for steps in range(1, len(masks)+1):
        total = 0
        for i in samples[0:nsamples]:
            b = np.zeros(n, dtype=np.float32)
            b[i] = 1.0
            for step in range(steps):
                b = (np.dot(b, masks[step]) > 0.0).astype(np.float32)
            total += np.sum(b)

        pct = 100.0 * total / (nsamples * n)
        factors.append("%.1f" % pct)
        if pct >= 99.99:
            break

    return " ".join(factors)