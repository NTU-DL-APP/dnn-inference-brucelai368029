import numpy as np

def relu(x):
    return np.maximum(0, x)
def softmax(x):
    x = np.array(x)
    # 支援一維和二維輸入
    if x.ndim == 1:
        x = x - np.max(x)
        e_x = np.exp(x)
        return e_x / np.sum(e_x)
    elif x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    else:
        raise ValueError("softmax only supports 1D or 2D arrays")
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
