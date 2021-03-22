import statistics


def quartiles(datapoints):
    sortedPoints = sorted(datapoints)
    mid = len(sortedPoints) // 2
    if len(sortedPoints) % 2 == 0:
        lowerQ = sortedPoints[:mid]
        higherQ = sortedPoints[mid:]
    else:
        lowerQ = sortedPoints[:mid]
        higherQ = sortedPoints[mid+1:]

    return median(lowerQ), median(sortedPoints), median(higherQ)


def median(datapoints):
    sortedPoints = sorted(datapoints)
    mid = len(sortedPoints) // 2
    if len(sortedPoints) % 2 == 0:
        med = sortedPoints[mid]
    else:
        med = (sortedPoints[mid] + sortedPoints[mid + 1]) / 2
    return med


def worst25(datapoints):
    sortedPoints = sorted(datapoints)
    sortedPoints = sortedPoints[int(0.75*len(datapoints)):]
    return sum(sortedPoints) / len(sortedPoints)


def best25(datapoints):
    sortedPoints = sorted(datapoints)
    sortedPoints = sortedPoints[:int(0.25*len(datapoints))]
    return sum(sortedPoints) / len(sortedPoints)


def trimean(datapoints):
    q1, q2, q3 = quartiles(datapoints)
    return (q1 + 2*q2 + q3) / 4


def variance(datapoints):
    return statistics.variance(datapoints)
