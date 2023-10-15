import tensorflow as tf

def piven_loss(lambda_in=15., soften=160., alpha=0.05, beta=0.5):
    """
    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """
    def piven_loss(y_true, y_pred):
        # from Figure 1 in the paper
        y_U = y_pred[:, 0] # U(x)
        y_L = y_pred[:, 1] # L(x)
        y_v = y_pred[:, 2] # v(x)
        y_T = y_true[:, 0] # y(x)

        N_ = tf.cast(tf.size(y_T), tf.float32)  # batch size
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = tf.multiply(tf.sigmoid((y_U - y_T) * soften),
                             tf.sigmoid((y_T - y_L) * soften))

        # k_hard uses sign step function
        k_hard = tf.multiply(tf.maximum(0., tf.sign(y_U - y_T)),
                             tf.maximum(0., tf.sign(y_T - y_L)))

        # MPIW_capt from equation 4
        MPIW_capt = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L) * k_hard),
                              tf.reduce_sum(k_hard) + 0.001)

        # equation 1 where k is k_soft
        PICP_soft = tf.reduce_mean(k_soft)

        # pi loss from section 4.2
        pi_loss =  MPIW_capt  + lambda_ * tf.sqrt(N_) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft))

        y_piven = y_v * y_U + (1 - y_v) * y_L # equation 3
        y_piven = tf.reshape(y_piven, (-1, 1))

        v_loss = tf.losses.mean_squared_error(y_true, y_piven)  # equation 5
        piven_loss_ = beta * pi_loss + (1-beta) * v_loss # equation 6

        return piven_loss_

    return piven_loss
