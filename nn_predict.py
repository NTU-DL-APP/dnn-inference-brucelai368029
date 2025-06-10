import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b

def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer.get('weights', [])

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

def nn_inference(model_arch, weights, data):
    data = np.array(data, dtype=np.float32)
    return nn_forward_h5(model_arch, weights, data)
