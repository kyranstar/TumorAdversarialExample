from abc import ABCMeta
import collections
import warnings
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans import utils
from cleverhans.attacks_tf import SPSAAdam, margin_logit_loss, UnrolledAdam
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.model import wrapper_warning, wrapper_warning_logits
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max
from cleverhans.utils_tf import clip_eta
from cleverhans import utils_tf
from cleverhans.attacks import Attack, optimize_linear


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Create a FastGradientMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(FastGradientMethod, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = ('eps', 'y', 'y_target', 'clip_min', 'clip_max', 'loss_func')
        self.structural_kwargs = ['ord', 'sanity_checks']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
        Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
        y_target=None if y is also set. Labels should be
        one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, _nb_classes = self.get_or_guess_labels(x, kwargs)

        return fgm(
            x,
            self.model.get_logits(x),
            y=labels,
            eps=self.eps,
            ord=self.ord,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            targeted=(self.y_target is not None),
            sanity_checks=self.sanity_checks,
            loss_func=self.loss_func)

    def parse_params(self,
        eps=0.3,
        ord=np.inf,
        y=None,
        y_target=None,
        clip_min=None,
        clip_max=None,
        sanity_checks=True,
        loss_func=None,
        **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
        Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
        y_target=None if y is also set. Labels should be
        one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param sanity_checks: bool, if True, include asserts
        (Turn them off to use less runtime / memory or for unit tests that
        intentionally pass strange input)
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sanity_checks = sanity_checks
        self.loss_func = loss_func

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
            # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True

def fgm(x,
        logits,
        y=None,
        eps=0.3,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False,
        sanity_checks=True,
        loss_func=None):
        """
        TensorFlow implementation of the Fast Gradient Method.
        :param x: the input placeholder
        :param logits: output of model.get_logits
        :param y: (optional) A placeholder for the model labels. If targeted
        is true, then provide the target label. Otherwise, only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param eps: the epsilon (input variation parameter)
        :param ord: (optional) Order of the norm (mimics NumPy).
        Possible values: np.inf, 1 or 2.
        :param clip_min: Minimum float value for adversarial example components
        :param clip_max: Maximum float value for adversarial example components
        :param targeted: Is the attack targeted or untargeted? Untargeted, the
        default, will try to make the label incorrect. Targeted
        will instead try to move in the direction of being more
        like y.
        :return: a tensor for the adversarial example
        """

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            asserts.append(utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

        if clip_max is not None:
            asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

        # Make sure the caller has not passed probs by accident
        assert logits.op.type != 'Softmax'

        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            preds_max = reduce_max(logits, 1, keepdims=True)
            y = tf.to_float(tf.equal(logits, preds_max))
            y = tf.stop_gradient(y)
        #y = y / tf.math.reduce_sum(y, 1, keepdims=True)

        # Compute loss
        loss = loss_func(labels=y, logits=logits)
        if targeted:
            loss = -loss

        # Define gradient of loss wrt input
        grad, = tf.gradients(loss, x)

        optimal_perturbation = optimize_linear(grad, eps, ord)

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            # We don't currently support one-sided clipping
            assert clip_min is not None and clip_max is not None
            adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

        if sanity_checks:
            with tf.control_dependencies(asserts):
                adv_x = tf.identity(adv_x)

        return adv_x

class Noise(Attack):
    """
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(Noise, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = ('eps', 'clip_min', 'clip_max', 'type')
        self.structural_kwargs = ['ord', 'sanity_checks']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
        Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
        y_target=None if y is also set. Labels should be
        one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        return add_noise(
            x,
            eps=self.eps,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            type=self.type)

    def parse_params(self,
        eps=0.3,
        clip_min=None,
        clip_max=None,
        type='Gaussian',
        **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
        Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
        y_target=None if y is also set. Labels should be
        one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param sanity_checks: bool, if True, include asserts
        (Turn them off to use less runtime / memory or for unit tests that
        intentionally pass strange input)
        """
        # Save attack-specific parameters
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.type = type

        return True

def add_noise(x,
        eps=0.3,
        clip_min=None,
        clip_max=None,
        type='Gaussian'):
        """
        :param x: the input placeholder
        :param logits: output of model.get_logits
        :param y: (optional) A placeholder for the model labels. If targeted
        is true, then provide the target label. Otherwise, only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param eps: the epsilon (input variation parameter)
        :param ord: (optional) Order of the norm (mimics NumPy).
        Possible values: np.inf, 1 or 2.
        :param clip_min: Minimum float value for adversarial example components
        :param clip_max: Maximum float value for adversarial example components
        :param targeted: Is the attack targeted or untargeted? Untargeted, the
        default, will try to make the label incorrect. Targeted
        will instead try to move in the direction of being more
        like y.
        :return: a tensor for the adversarial example
        """

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            asserts.append(utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

        if clip_max is not None:
            asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

        if type == 'Gaussian':
            perturbation = tf.random.normal(
                                        x.shape,
                                        mean=0.0,
                                        stddev=eps)
        elif type == 'Uniform':
            perturbation = tf.random.uniform(x.shape,
                                        minval=-eps,
                                        maxval=eps)
        else:
            print("Unknown noise type")

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + perturbation

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            # We don't currently support one-sided clipping
            assert clip_min is not None and clip_max is not None
            adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

        return adv_x


class ProjectedGradientDescent(Attack):
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  """

  FGM_CLASS = FastGradientMethod

  def __init__(self, model, sess=None, dtypestr='float32',
               default_rand_init=True, **kwargs):
    """
    Create a ProjectedGradientDescent instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(ProjectedGradientDescent, self).__init__(model, sess=sess,
                                                   dtypestr=dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                            'clip_max', 'loss_func')
    self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'sanity_checks']
    self.default_rand_init = default_rand_init

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param rand_init: (optional) Whether to use random initialization
    :param y: (optional) A tensor with the true class labels
      NOTE: do not use smoothed labels here
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
      NOTE: do not use smoothed labels here
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    # Initialize loop variables
    if self.rand_init:
      eta = tf.random_uniform(tf.shape(x),
                              tf.cast(-self.rand_minmax, x.dtype),
                              tf.cast(self.rand_minmax, x.dtype),
                              dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, self.ord, self.eps)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    if self.y_target is not None:
      y = self.y_target
      targeted = True
    elif self.y is not None:
      y = self.y
      targeted = False
    else:
      model_preds = self.model.get_probs(x)
      preds_max = reduce_max(model_preds, 1, keepdims=True)
      y = tf.to_float(tf.equal(model_preds, preds_max))
      y = tf.stop_gradient(y)
      targeted = False
      del model_preds

    y_kwarg = 'y_target' if targeted else 'y'
    fgm_params = {
        'eps': self.eps_iter,
        y_kwarg: y,
        'ord': self.ord,
        'clip_min': self.clip_min,
        'clip_max': self.clip_max,
        'loss_func': self.loss_func
    }
    if self.ord == 1:
      raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                " step for PGD when ord=1, because ord=1 FGM "
                                " changes only one pixel at a time. We need "
                                " to rigorously test a strong ord=1 PGD "
                                "before enabling this feature.")

    # Use getattr() to avoid errors in eager execution attacks
    FGM = self.FGM_CLASS(
        self.model,
        sess=getattr(self, 'sess', None),
        dtypestr=self.dtypestr)

    def cond(i, _):
      return tf.less(i, self.nb_iter)

    def body(i, adv_x):
      #fgm_params['loss_func'] = self.loss_func#(labels=fgm_params['y'], logits=self.model.get_logits(adv_x))
      adv_x = FGM.generate(adv_x, **fgm_params)

      # Clipping perturbation eta to self.ord norm ball
      eta = adv_x - x
      eta = clip_eta(eta, self.ord, self.eps)
      adv_x = x + eta

      # Redo the clipping.
      # FGM already did it, but subtracting and re-adding eta can add some
      # small numerical error.
      if self.clip_min is not None or self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      return i + 1, adv_x

    _, adv_x = tf.while_loop(cond, body, [tf.zeros([]), adv_x], back_prop=True)

    asserts = []

    # Asserts run only on CPU.
    # When multi-GPU eval code tries to force all PGD ops onto GPU, this
    # can cause an error.
    with tf.device("/CPU:0"):
      asserts.append(tf.assert_less_equal(self.eps_iter, self.eps))
      if self.ord == np.inf and self.clip_min is not None:
        # The 1e-6 is needed to compensate for numerical error.
        # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5,
        # clip_max=.7
        asserts.append(tf.assert_less_equal(self.eps,
                                            1e-6 + self.clip_max
                                            - self.clip_min))

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

  def parse_params(self,
                   eps=0.3,
                   eps_iter=0.05,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   rand_init=None,
                   rand_minmax=0.3,
                   sanity_checks=True,
                   loss_func=None,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    Attack-specific parameters:
    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the model labels.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool Insert tf asserts checking values
        (Some tests need to run with no sanity checks because the
         tests intentionally configure the attack strangely)
    """

    # Save attack-specific parameters
    self.eps = eps
    if rand_init is None:
      rand_init = self.default_rand_init
    self.rand_init = rand_init
    if self.rand_init:
      self.rand_minmax = eps
    else:
      self.rand_minmax = 0.
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.loss_func = loss_func

    if isinstance(eps, float) and isinstance(eps_iter, float):
      # If these are both known at compile time, we can check before anything
      # is run. If they are tf, we can't check them yet.
      assert eps_iter <= eps, (eps_iter, eps)

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")
    self.sanity_checks = sanity_checks

    return True


class MomentumIterativeMethod(Attack):
    """
    The Momentum Iterative Method (Dong et al. 2017). This method won
    the first places in NIPS 2017 Non-targeted Adversarial Attacks and
    Targeted Adversarial Attacks. The original paper used hard labels
    for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Create a MomentumIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(MomentumIterativeMethod, self).__init__(model, sess, dtypestr,
                                                      **kwargs)
        self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                                'clip_max', 'loss_func')
        self.structural_kwargs = ['ord', 'nb_iter', 'decay_factor', 'sanity_checks']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param kwargs: Keyword arguments. See `parse_params` for documentation.
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)
        asserts = []

        # If a data range was specified, check that the input was in that range
        if self.clip_min is not None:
            asserts.append(utils_tf.assert_greater_equal(x,
                                                       tf.cast(self.clip_min,
                                                               x.dtype)))

        if self.clip_max is not None:
            asserts.append(utils_tf.assert_less_equal(x,
                                                    tf.cast(self.clip_max,
                                                            x.dtype)))

        # Initialize loop variables
        momentum = tf.zeros_like(x)
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        y, _nb_classes = self.get_or_guess_labels(x, kwargs)
        #y = y / reduce_sum(y, 1, keepdims=True)
        targeted = (self.y_target is not None)

        def cond(i, _, __):
            return tf.less(i, self.nb_iter)

        def body(i, ax, m):
            logits = self.model.get_logits(ax)
            loss = self.loss_func(labels=y, logits=logits)
            if targeted:
                loss = -loss

            # Define gradient of loss wrt input
            grad, = tf.gradients(loss, ax)

            # Normalize current gradient and add it to the accumulated gradient
            red_ind = list(xrange(1, len(grad.get_shape())))
            avoid_zero_div = tf.cast(1e-12, grad.dtype)
            grad = grad / tf.maximum(
              avoid_zero_div,
              reduce_mean(tf.abs(grad), red_ind, keepdims=True))
            m = self.decay_factor * m + grad

            optimal_perturbation = optimize_linear(m, self.eps_iter, self.ord)
            if self.ord == 1:
                raise NotImplementedError("This attack hasn't been tested for ord=1."
                                          "It's not clear that FGM makes a good inner "
                                          "loop step for iterative optimization since "
                                          "it updates just one coordinate at a time.")

            # Update and clip adversarial example in current iteration
            ax = ax + optimal_perturbation
            ax = x + utils_tf.clip_eta(ax - x, self.ord, self.eps)

            if self.clip_min is not None and self.clip_max is not None:
                ax = utils_tf.clip_by_value(ax, self.clip_min, self.clip_max)

            ax = tf.stop_gradient(ax)

            return i + 1, ax, m

        _, adv_x, _ = tf.while_loop(
            cond, body, [tf.zeros([]), adv_x, momentum], back_prop=True)

        if self.sanity_checks:
          with tf.control_dependencies(asserts):
            adv_x = tf.identity(adv_x)

        return adv_x

    def parse_params(self,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   decay_factor=1.0,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   loss_func=None,
                   **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.decay_factor = decay_factor
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sanity_checks = sanity_checks
        self.loss_func = loss_func

        if self.y is not None and self.y_target is not None:
          raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
          raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True
