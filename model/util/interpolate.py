def interpolate(t, degree, points, knots=None, weights=None, result=[]):
    n = len(points)
    d = len(points[0])

    if (degree < 1 or degree > n-1):
        raise ValueError('Degree must be at least 1 and less than or equal to point count - 1')
    if weights is None:
        weights = []
        for i in range(n):
            weights.append(float(1))
    if knots is None:
        knots = []
        for i in range(n+degree+1):
            knots.append(float(i))
    else:
        if len(knots) != n + degree + 1:
            raise Exception('bad knot vector length')
    domain = [degree, len(knots) - 1 - degree]
    low = knots[domain[0]]
    high = knots[domain[1]]
    t = t * (high - low) + low

    if (t < low or t > high):
        raise Exception("Out of bounds")
    
    for s in range(domain[0], domain[1]):
        if (t >= knots[s] and t <= knots[s+1]):
            break

    v = []
    for i in range(n):
        j_list = [points[i][j] * weights[i] for j in range(d)]
        v.append(j_list)
        v[i].append(weights[i])

    for l in range(1, degree+2):
        for i in range(s, s-degree-1+l, -1):
            alpha = (t - knots[i]) / (knots[i+degree+1-l] - knots[i])

            for j in range(d+1):
                v[i][j] = (1 - alpha) * v[i-1][j] + alpha * v[i][j]
    
    result = [v[s][i] / v[s][d] for i in range(d)]
    return result