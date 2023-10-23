import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_theta = 0.5 + 0.5 * cos_theta
    return cos_theta

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_distance = 1 - cos_theta
    # cos_distance = 0.5 - 0.5 * cos_theta
    return cos_distance