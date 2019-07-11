import math

import numpy as np


def adjoint_test1(A, AT, x, y, tolerance=0.01):
    l = np.dot(y.flat, np.matmul(A, x.flat).flat)
    r = np.dot(np.matmul(AT, y.flat).flat, x.flat)

    if r * (1 + tolerance) > l > r * (1 - tolerance):
        print("Passes adjoint test 1 using a tolerance of {0}%".format(tolerance * 100))
        return True
    else:
        return False


def adjoint_test2(x, y , x_construct, y_construct, tolerance=0.01):
    l = np.dot(x.flat, x_construct.flat)
    r = np.dot(y_construct.flat, y.flat)

    if r * (1 + tolerance) > l > r * (1 - tolerance):
        print("Passes adjoint test 2 using a tolerance of {0}%".format(tolerance * 100))
        return True
    else:
        return False


def check_max_seg_difference(nRings, max_seg_diff):
    if max_seg_diff is None:
        # Assume all full max_seg_difference
        return nRings - 1
    elif max_seg_diff > nRings - 1:
        exit("ERROR: max_seg_diff > nRings - 1. nRings = {0}, max_seg_diff = {1}".format(nRings, max_seg_diff))
    else:
        return max_seg_diff


def upsample_size(nRings, max_seg_diff=None):
    max_seg_diff = check_max_seg_difference(nRings, max_seg_diff)

    counter = 0
    for i in range(nRings):
        for j in range(i, nRings):
            if abs(i - j) > max_seg_diff:
                break
            counter = counter + 1
    return counter


def gen_upsample_transformation_matrix(nRings, max_seg_diff=None):
    max_seg_diff = check_max_seg_difference(nRings, max_seg_diff)
    A = np.zeros([upsample_size(nRings, max_seg_diff), nRings])

    current_ring = 0
    reaching_ring = 0

    for i in range(upsample_size(nRings, max_seg_diff)):
        A[i, current_ring] = A[i, current_ring] + 1
        A[i, reaching_ring] = A[i, reaching_ring] + 1

        reaching_ring = reaching_ring + 1
        if reaching_ring == nRings or abs(current_ring - reaching_ring) > max_seg_diff:
            current_ring = current_ring + 1
            reaching_ring = current_ring
            if reaching_ring == nRings:
                break
    return A / 2


def gen_downsample_transformation_matrix(nRings, max_seg_diff=None):
    max_seg_diff = check_max_seg_difference(nRings, max_seg_diff)
    A = np.zeros([nRings, upsample_size(nRings, max_seg_diff)])

    current_ring = 0
    reaching_ring = 0

    for i in range(upsample_size(nRings, max_seg_diff)):
        A[current_ring, i] = A[current_ring, i] + 1
        A[reaching_ring, i] = A[reaching_ring, i] + 1

        reaching_ring = reaching_ring + 1
        if reaching_ring == nRings or abs(current_ring - reaching_ring) > max_seg_diff:
            current_ring = current_ring + 1
            reaching_ring = current_ring
            if reaching_ring == nRings:
                break

    return A / 2


def upscale_no_transformation_matrix(x, nRings, max_seg_diff=None):

    max_seg_diff = check_max_seg_difference(nRings, max_seg_diff)
    y = np.zeros([upsample_size(nRings, max_seg_diff), 1])

    yi = 0
    for xi_current in range(x.size):
        for xi_reaching in range(xi_current, x.size):
            if xi_reaching == nRings or abs(xi_current - xi_reaching) > max_seg_diff:
                break

            y[yi] = x[xi_current] + x[xi_reaching]
            yi += 1

    return 0.5 * y


def downscale_no_transformation_matrix(y, nRings, max_seg_diff=None):
    max_seg_diff = check_max_seg_difference(nRings, max_seg_diff)

    x = np.zeros([nRings, 1])

    yi = 0

    for xi_current in range(x.size):
        for xi_reaching in range(xi_current, x.size):

            if xi_reaching == nRings or abs(xi_current - xi_reaching) > max_seg_diff:
                break

            x[xi_current] += y[yi]
            x[xi_reaching] += y[yi]
            yi +=1

    return x/2



if __name__ == "__main__":
    nRings = 1000

    max_seg_diff = None

    # Define downscaled data x and upscaled data y
    x = np.random.rand(nRings, 1)
    # x = np.ones([nRings, 1])
    #
    y = np.random.rand(upsample_size(nRings, max_seg_diff), 1)
    # y = np.ones([upsample_size(nRings, max_seg_diff), 1])

    # A = np.eye(np.size(y), np.size(x))

    A = gen_upsample_transformation_matrix(nRings, max_seg_diff)
    AT = gen_downsample_transformation_matrix(nRings, max_seg_diff)
    adjoint_test1(A, AT, x, y)

    y_construct = upscale_no_transformation_matrix(x, nRings, max_seg_diff)
    x_construct = downscale_no_transformation_matrix(y, nRings, max_seg_diff)

    y_test = np.matmul(A, x)
    x_test = np.matmul(AT, y)

    adjoint_test2(x, y, x_construct, y_construct)






