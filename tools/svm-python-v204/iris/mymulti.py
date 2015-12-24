"""A module that SVM^python interacts with to do its evil bidding."""

import svmapi
import pdb
import numpy as np

def read_examples(filename, sparm):
    """Reads and returns x,y example pairs from a file.
    
    This reads the examples contained at the file at path filename and
    returns them as a sequence.  Each element of the sequence should
    be an object 'e' where e[0] and e[1] is the pattern (x) and label
    (y) respectively.  Specifically, the intention is that the element
    be a two-element tuple containing an x-y pair."""
    # This reads example files of the type read by SVM^multiclass.
    examples = []
    with open(filename) as f:
        for line in f:
            if line.find('#'): line = line[:line.find('#')]

            tokens = line.split()
            if not tokens: continue

            target = int(tokens[0])
            tokens = [t.split(':') for t in tokens[1:]]
            features = [(0,1)]+[(int(k),float(v)) for k,v in tokens]
            examples.append((svmapi.Sparse(features), target))

    print len(examples),'examples read'
    return examples

def init_model(sample, sm, sparm):
    """Initializes the learning model.
    
    Initialize the structure model sm.  The sm.size_psi must be set to
    the number of features.  The ancillary purpose is to add any
    information to sm that is necessary from the user code
    perspective.  This function returns nothing."""
    # In our binary classification task, we've encoded a pattern as a
    # list of four features.  We just want a linear rule, so we have a
    # weight corresponding to each feature.  We also add one to allow
    # for a last "bias" feature.

    sm.num_features = max(max(x) for x,y in sample)[0]+1
    sm.num_classes = max(y for x,y in sample)
    sm.size_psi = sm.num_features*sm.num_classes

def init_constraints(sample, sm, sparm):
    """Initializes special constraints.

    Returns a sequence of initial constraints.  Each constraint in the
    returned sequence is itself a sequence with two items (the
    intention is to be a tuple).  

    [(document object1, constraint1), 
     (document object2, constraint2), 
     (document object3, constraint3)] 

    1. The first item of the tuple is a document object.  
    2. The second item is a number, indicating that the inner product of
    the feature vector of the document object with the linear weights must 
    be greater than or equal to the number (or, in the nonlinear case, the 
    evaluation of the kernel on the feature vector with the current model 
    must be greater).  
    
    This initializes the optimization problem by allowing the introduction
    of special constraints.  Typically no special constraints are
    necessary.  A typical constraint may be to ensure that all feature
    weights are positive.

    Note that the slack id must be set. The slack IDs 1 through
    len(sample) (or just 1 in the combined constraint option) are used
    by the training examples in the sample, so do not use these if you
    do not intend to share slack with the constraints inferred from
    the training data.

    The default behavior is equivalent to returning an empty list,
    i.e., no constraints."""

    if False:
        # Just some example cosntraints.
        c, d = svmapi.Sparse, svmapi.Document
        # Return some really goofy constraints!  Normally, if the SVM
        # is allowed to converge normally, the second and fourth
        # features are 0 and -1 respectively for sufficiently high C.
        # Let's make them be greater than 1 and 0.2 respectively!!
        # Both forms of a feature vector (sparse and then full) are
        # shown.
        return [
                (
                    svmapi.Document
                    (
                        [
                            c([(1,1)]) # equal to [0, 1, 0, 0, 0]
                        ],
                        slackid=len(sample)+1
                    ),
                    1 # <W, [0, 1, 0, 0, 0]> >= 1.0 - slack(len(sample)+1)
                ),

                (
                    svmapi.Document
                    (
                        [
                            c([0,0,0,1])
                        ],
                        slackid=len(sample)+1
                    ),
                    .2 # <W, [0, 0, 0, 1, 0]> >= 0.2 - slack(len(sample)+1)
                )
               ]
        # c = 1.0       [0.5, 0.166667, 0, -0.5, 0]
        # c = 1000.0    [3.2, 1, 2.22045e-16, 0.2, -2.2]



    # Encode positivity constraints.  Note that this constraint is
    # satisfied subject to slack constraints.
#   constraints = []
#   for i in xrange(sm.size_psi):
#       # Create a sparse vector which selects out a single feature.
#       sparse = svmapi.Sparse([(i,1)])
#       # The left hand side of the inequality is a document.
#       lhs = svmapi.Document([sparse], costfactor=1, slackid=i+1)
#       # Append the lhs and the rhs (in this case 0).
#       constraints.append((lhs, 0.0))
#   return constraints


    constraints = []
    slack_count = 0
    for x, y in sample:
        for ybar in xrange(1, sm.num_classes+1):
            if y == ybar: continue
            slack_count += 1
            lhs = svmapi.Document([svmapi.Sparse(diff_psi(x, y, ybar, sm, sparm))], slackid=slack_count)
            rhs = 1.0
            constraints.append((lhs, rhs))

    print('|Y| = {}, N = {}, (|Y|-1)*N = {}'.format(sm.num_classes, len(sample), (sm.num_classes-1)*len(sample)) )
    print('Count of constraints: {} (press to continue)'.format( len(constraints) ) )
    return constraints

def sparse2array(sparse, l = None):
    if not l:
        l = max(sparse)[0]
    arr = [0] * l
    for ind, val in sparse:
        arr[ind] = val
    return arr

def classification_score(x, y, sm, sparm):
# passed
#   ybar = 1
#   print 'y = {}, ybar = {}'.format(y, ybar)
#   print '<W, psi(x, y)> =', np.dot(sm.w[:], sparse2array(psi(x, y, sm, sparm)[:], sm.size_psi))
#   print '<W, psi(x, ybar)> =', np.dot(sm.w[:], sparse2array(psi(x, ybar, sm, sparm)[:], sm.size_psi))
#   print '<W, diff_psi(x, y, ybar)> =', np.dot(sm.w[:], sparse2array(diff_psi(x, y, ybar, sm, sparm)[:], sm.size_psi))
#   pdb.set_trace() 
    score = 0
    for ind, ele in x:
        try:
            score += sm.w[sm.num_features*(y-1)+ind] * ele
        except:
            pdb.set_trace()

    return score

def classify_example(x, sm, sparm):
    """Given a pattern x, return the predicted label."""
    # Believe it or not, this is a dot product.  The last element of
    # sm.w is assumed to be the weight associated with the bias
    # feature as explained earlier.

    scores = [(classification_score(x,c,sm,sparm), c) for c in xrange(1,sm.num_classes+1)]
    return max(scores)[1] 

def find_most_violated_constraint(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    Returns the label ybar for pattern x corresponding to the most
    violated constraint according to SVM^struct cost function.  To
    find which cost function you should use, check sparm.loss_type for
    whether this is slack or margin rescaling (1 or 2 respectively),
    and check sparm.slack_norm for whether the slack vector is in an
    L1-norm or L2-norm in the QP (1 or 2 respectively).

    If this function is not implemented, this function is equivalent
    to 'classify(x, sm, sparm)'.  The optimality guarantees of
    Tsochantaridis et al. no longer hold, since this doesn't take the
    loss into account at all, but it isn't always a terrible
    approximation.  One still technically maintains the empirical
    risk bound condition, but without any regularization."""

    scores = [(
                loss(y,ypred,sparm) - ( classification_score(x, y, sm, sparm)-classification_score(x, ypred, sm, sparm) ),
                ypred
             )
                for ypred in xrange(1,sm.num_classes+1)
            ]
    ybar = max(scores)[1]
    #print y, ybar
    return max(scores)[1]

def find_most_violated_constraint_slack(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for slack rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def find_most_violated_constraint_margin(x, y, sm, sparm):
    """Return ybar associated with x's most violated constraint.

    The find most violated constraint function for margin rescaling.
    The default behavior is that this returns the value from the
    general find_most_violated_constraint function."""
    return find_most_violated_constraint(x, y, sm, sparm)

def psi(x, y, sm, sparm):
    """Return a feature vector representing pattern x and label y.

    This is the combined feature function, which this returns either a
    svmapi.Sparse object, or sequence of svmapi.Sparse objects (useful
    during kernel evaluations, as all components undergo kernel
    evaluation separately).  There is no default behavior."""
    # In the case of binary classification, psi is just the class (+1
    # or -1) times the feature vector for x, including that special
    # constant bias feature we pretend that we have.
    jf = map(lambda ele: (ele[0]+sm.num_features*(y-1), ele[1]), x)
    return svmapi.Sparse(jf)

def diff_psi(x, y, ybar, sm, sparm):
    if y == ybar: return []
    if y < ybar:
        jf = map(lambda ele: (ele[0]+sm.num_features*(y-1), ele[1]), x) + \
             map(lambda ele: (ele[0]+sm.num_features*(ybar-1), -ele[1]), x) 
    else:
        jf = map(lambda ele: (ele[0]+sm.num_features*(ybar-1), -ele[1]), x) + \
             map(lambda ele: (ele[0]+sm.num_features*(y-1), ele[1]), x) 
    return jf

def loss(y, ybar, sparm):
    """Return the loss of ybar relative to the true labeling y.
    
    Returns the loss for the correct label y and the predicted label
    ybar.  In the event that y and ybar are identical loss must be 0.
    Presumably as y and ybar grow more and more dissimilar the
    returned value will increase from that point.  sparm.loss_function
    holds the loss function option specified on the command line via
    the -l option.

    The default behavior is to perform 0/1 loss based on the truth of
    y==ybar."""
    # If they're the same sign, then the loss should be 0.
    return 100.0*int(y != ybar)

def print_iteration_stats(ceps, cached_constraint, sample, sm,
                          cset, alpha, sparm):
    """Called just before the end of each cutting plane iteration.

    This is called just before the end of each cutting plane
    iteration, primarily to print statistics.  The 'ceps' argument is
    how much the most violated constraint was violated by.  The
    'cached_constraint' argument is true if this constraint was
    constructed from the cache.
    
    The default behavior is that nothing is printed."""
    print

def print_learning_stats(sample, sm, cset, alpha, sparm):
    """Print statistics once learning has finished.
    
    This is called after training primarily to compute and print any
    statistics regarding the learning (e.g., training error) of the
    model on the training sample.  You may also use it to make final
    changes to sm before it is written out to a file.  For example, if
    you defined any non-pickle-able attributes in sm, this is a good
    time to turn them into a pickle-able object before it is written
    out.  Also passed in is the set of constraints cset as a sequence
    of (left-hand-side, right-hand-side) two-element tuples, and an
    alpha of the same length holding the Lagrange multipliers for each
    constraint.

    The default behavior is that nothing is printed."""
    print 'Model learned:',
    print '[',', '.join(['%g'%i for i in sm.w]),']'
    print 'Losses:',
    print [loss(y, classify_example(x, sm, sparm), sparm) for x,y in sample]

def print_testing_stats(sample, sm, sparm, teststats):
    """Print statistics once classification has finished.
    
    This is called after all test predictions are made to allow the
    display of any summary statistics that have been accumulated in
    the teststats object through use of the eval_prediction function.

    The default behavior is that nothing is printed."""
    print teststats

def eval_prediction(exnum, (x, y), ypred, sm, sparm, teststats):
    """Accumulate statistics about a single training example.
    
    Allows accumulated statistics regarding how well the predicted
    label ypred for pattern x matches the true label y.  The first
    time this function is called teststats is None.  This function's
    return value will be passed along to the next call to
    eval_prediction.  After all test predictions are made, the last
    value returned will be passed along to print_testing_stats.

    On the first call, that is, when exnum==0, teststats==None.  The
    default behavior is that the function does nothing."""
    #if exnum==0: teststats = []
    #print 'on example',exnum,'predicted',ypred,'where correct is',y
    #teststats.append(loss(y, ypred, sparm))
    return teststats

def write_model(filename, sm, sparm):
    """Dump the structmodel sm to a file.
    
    Write the structmodel sm to a file at path filename.

    The default behavior is equivalent to
    'cPickle.dump(sm,bz2.BZ2File(filename,'w'))'."""
    import cPickle, bz2
    f = bz2.BZ2File(filename, 'w')
    cPickle.dump(sm, f)
    f.close()

def read_model(filename, sparm):
    """Load the structure model from a file.
    
    Return the structmodel stored in the file at path filename, or
    None if the file could not be read for some reason.

    The default behavior is equivalent to
    'return cPickle.load(bz2.BZ2File(filename))'."""
    import cPickle, bz2
    return cPickle.load(bz2.BZ2File(filename))

def write_label(fileptr, y):
    """Write a predicted label to an open file.

    Called during classification, this function is called for every
    example in the input test file.  In the default behavior, the
    label is written to the already open fileptr.  (Note that this
    object is a file, not a string.  Attempts to close the file are
    ignored.)  The default behavior is equivalent to
    'print>>fileptr,y'"""
    print>>fileptr,y

def print_help():
    """Help printed for badly formed CL-arguments when learning.

    If this function is not implemented, the program prints the
    default SVM^struct help string as well as a note about the use of
    the --m option to load a Python module."""
    import svmapi
    print svmapi.default_help
    print "This is a help string for the learner!"

def print_help_classify():
    """Help printed for badly formed CL-arguments when classifying.

    If this function is not implemented, the program prints the
    default SVM^struct help string as well as a note about the use of
    the --m option to load a Python module."""
    print "This is a help string for the classifer!"
