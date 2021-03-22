import statistics_utils as stats
import tensorflow as tf


def error_statistics(stat):
    metric = tf.convert_to_tensor(stat)
    mean = tf.reduce_mean(metric)
    std = tf.math.reduce_std(metric)
    median = stats.median(stat)
    trimean = stats.trimean(stat)
    max = tf.reduce_max(metric)
    argmax = tf.argmax(metric)
    min = tf.reduce_min(metric)
    argmin = tf.argmin(metric)
    best25 = stats.best25(metric)
    worst25 = stats.worst25(metric)

    report = {'mean': mean, 'std': std, 'median': median, 'trimean': trimean,
              'max': max, 'argmax': argmax, 'min': min, 'argmin': argmin, 'best25': best25, 'worst25': worst25}
    return report


def report_to_string(report):
    return f'{report["mean"]:.2f}\t {report["std"]:.2f}\t {report["median"]:.2f}\t {report["trimean"]:.2f}\t {report["max"]:.2f} ({report["argmax"]})\t {report["min"]:.2f} ({report["argmin"]})\t {report["worst25"]:.2f}\t {report["best25"]:.2f}'

def print_report(report):
    print(report_to_string(report))

def write_report(report, f, name=None):
    if name is not None:
        f.write(name)
        f.write(" ")
    f.write(report_to_string(report))
    f.write('\n')