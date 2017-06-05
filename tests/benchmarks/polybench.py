# mini or extra large datasets

dictionary = {
    # alpha = 1.5
    # beta = 1.2
    # N is 22 or 2200
    '2mm_1': {
        'e': 't + (1.5 * a * b)',
        # N * alpha * max(a*b)
        'v': {'a':[0,1],'b':[0,1],'t':[0,2200*1.5*1],} # or N=22
    },
    # N is 18 or 1800
    '2mm_2': {
        'e': 'd + (t * c)',
        # d initially in beta*[0,1]
        # N * max(t*c) + d
        'v': {'t':[0,1],'c':[0,1],'d':[0,1800*(2200*1.5*1)+1.2],} # or N=18
    },
    # N = 20 or 2000
    '3mm': {
        'e': 'e + (a * b)',
        # N * max(a*b)
        'v': {'a':[0,0.2],'b':[0,0.2],'e':[0,2000*0.04],} # or N=20
    },
    # NX is 20 or 2000
    # NY is 30 or 2600
    # TMAX is 20 or 1000
    'fdtd_1': {
        'e': 'a + (0.5 * (c + b))',
        'v': {'a':[0,1],'b':[-1,0],'c':[0,1]}
    },
    # N * 
    'fdtd': {
        'e': 'h + (-0.7)*(e + f + y + z)',
        'v': {'h':[0,1], 'e':[0,1], 'f':[-1,0], 'y':[0,1], 'z':[-1,0], 'h':[0,1],}
    },
    'gemm': {
        'e': 'C + 32412 * A * B',
        'v': {
            'A': [0, 1],
            'B': [0, 1],
            'C': [0, 1]
        }
    },
    'seidel': {
        'e': '0.2*(a+b+c+d+e)',
        'v': {
            'a': [0, 1],
            'b': [0, 1],
            'c': [0, 1],
            'd': [0, 1],
            'e': [0, 1],
        }
    },
    'symm': {
        'e': 'beta * C + alpha * A * B + alpha * acc',
        'v': { # find ranges
            'alpha': [0, 1],
            'beta': [0, 1],
            'A': [0, 1],
            'B': [0, 1],
            'C': [0, 1],
            'acc': [0, 1],
        }
    },
    # alpha = 1.5
    # beta = 1.2
    # M is 20 or 2000
    # N is 30 or 2600
    'syr2k': {
        'e': 'c + (a*1.5*b) + (e*1.5*d)',
        'v': {
            'a': [0, 1],
            'b': [0, 1],
            'c': [0, 1.2*2600.0/2000],
            'd': [0, 1],
            'e': [0, 1],
        }
    },
    # alpha = 1.5
    # beta = 1.2
    'syrk': {
        'e': 'c + (1.5*a*b)',
        'v': {
            'a': [0, 1],
            'b': [0, 1],
            'c': [0, 1.2],
        }
    },
    
}