import joblib
import os
import tensorflow as tf
from sagui.utils.logx import restore_tf_graph

def load_policy_transfer(fpath, itr='last'):

    # handle which epoch to load from
    if itr=='last':
        # saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session(graph=tf.Graph())

    # model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))
    model = restore_tf_graph(sess, os.path.join(fpath, 'tf1_save'+itr))

    # get the correct op for executing actions
    
    log_action_op = model['logp_a']
    action_op = model['pi']

    # make function for producing an action given a single state
    get_logp_a = lambda x, a : sess.run(log_action_op, feed_dict={model['x']: x[None,:], model['a']: a[None,:]})[0]
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(os.path.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_logp_a, get_action, sess
