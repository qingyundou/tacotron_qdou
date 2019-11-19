import numpy as np
def log_gradients(sess, model, log):
    [grads] = sess.run([model.gradients_fetchable])
    log('Inf / Nan found in:')
    for i,g in enumerate(grads):
        flag_inf = np.any(np.isinf(g)) if isinstance(g,np.ndarray) else np.any(np.isinf(g.values))
        flag_nan = np.any(np.isnan(g)) if isinstance(g,np.ndarray) else np.any(np.isnan(g.values))
        if flag_inf:
            log('inf, index %s, grad %s' % (str(i), model.gradients_fetchable[i].name))
        if flag_nan:
            log('nan, index %s, grad %s' % (str(i), model.gradients_fetchable[i].name))
